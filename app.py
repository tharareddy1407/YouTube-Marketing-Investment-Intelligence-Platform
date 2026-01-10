import os
import base64
from pathlib import Path
from datetime import datetime, timezone
import re

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build

# ==================================================
# Page Config (must be first Streamlit call)
# ==================================================
st.set_page_config(
    page_title="YouTube Marketing Investment Intelligence Platform",
    layout="wide"
)

# ==================================================
# Background Image (Local file, Render-safe)
# ==================================================
def set_local_background(image_path: str):
    img_path = Path(image_path)
    if not img_path.exists():
        st.warning(f"Background image not found at: {image_path}. App will load without background.")
        return

    encoded = base64.b64encode(img_path.read_bytes()).decode()
    ext = img_path.suffix.lower().replace(".", "") or "png"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 Put your background image here
set_local_background("assets/background.png")

# ==================================================
# UI Styling
# ==================================================
st.markdown("""
<style>
.content-box {
    background: rgba(255, 255, 255, 0.90);
    padding: 1.75rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.08);
}
.small-note {
    font-size: 0.85rem;
    color: #555;
    margin-top: -0.5rem;
}
.metric-pill {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(0,0,0,0.06);
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Region -> Language options (Dynamic)
# ==================================================
REGION_LANGUAGE_MAP = {
    "USA": ["English", "Spanish"],
    "India": [
        "Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam",
        "Marathi", "Bengali", "Gujarati", "Punjabi", "Urdu", "Odia"
    ],
    "UK": ["English", "Welsh", "Scottish Gaelic", "Irish"],
    "Canada": ["English", "French"],
    "Australia": ["English"],
    "UAE": ["Arabic", "English", "Hindi", "Urdu"],
    "Singapore": ["English", "Mandarin", "Malay", "Tamil"],
}

REGION_CODE_MAP = {
    "USA": "US",
    "India": "IN",
    "UK": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "UAE": "AE",
    "Singapore": "SG",
}

# ==================================================
# Helpers
# ==================================================
def safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

def days_ago(iso_date: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 9999

def tokenize(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(w for w in text.split() if len(w) > 2)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def compute_match_score(keywords, channel_text, video_titles, subs, avg_views, uploads_90d, inactive_days, geo_lang_conf):
    kw_tokens = set()
    for kw in keywords:
        kw_tokens |= tokenize(kw)

    ch_tokens = tokenize(channel_text)
    vid_tokens = set()
    for t in video_titles:
        vid_tokens |= tokenize(t)

    # Relevance (0–40)
    hits_channel = len(kw_tokens & ch_tokens)
    hits_videos = len(kw_tokens & vid_tokens)
    relevance = clamp((hits_channel * 2) + (hits_videos * 1), 0, 40)

    # Engagement (0–25) based on avg_views/subs
    ratio = (avg_views / max(subs, 1)) if subs else 0.0
    engagement = 25 if ratio >= 0.30 else clamp((ratio / 0.30) * 25, 0, 25)

    # Activity (0–20) using uploads in last 90 days (12+ => full)
    activity = clamp((uploads_90d / 12.0) * 20, 0, 20)

    # Geo/Lang confidence (0–10)
    geo_lang = clamp(geo_lang_conf, 0, 10)

    # Penalties (-15..0)
    penalty = 0
    if inactive_days > 120:
        penalty -= 10
    if ratio < 0.02:
        penalty -= 5

    total = clamp(relevance + engagement + activity + geo_lang + penalty, 0, 100)

    return total, relevance, engagement, activity, geo_lang, penalty, ratio

def infer_geo_lang_confidence(region, language, channel_title, channel_desc, video_titles):
    # Simple MVP heuristic (0–10)
    text = " ".join([channel_title or "", channel_desc or ""] + (video_titles or [])).lower()
    score = 0

    # Region signals
    if region.lower() in text:
        score += 4
    if region == "USA" and any(s in text for s in ["usa", "united states", "u.s.", "america"]):
        score += 4
    if region == "India" and any(s in text for s in ["india", "bharat", "indian"]):
        score += 4

    # Language signals (very basic)
    if language.lower() in text:
        score += 2

    return clamp(score, 0, 10)

# ==================================================
# YouTube API
# ==================================================
API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ Missing YOUTUBE_API_KEY. Add it as an environment variable (local terminal or Render).")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

# ==================================================
# Sidebar Inputs
# ==================================================
with st.sidebar:
    st.header("🔍 Campaign Inputs")

    region = st.selectbox("Region", list(REGION_LANGUAGE_MAP.keys()), index=0)
    region_code = REGION_CODE_MAP.get(region, "US")

    language_options = REGION_LANGUAGE_MAP.get(region, ["English"])
    language = st.selectbox("Language", language_options, index=0)

    goal = st.radio(
        "Investment Goal",
        ["Brand Awareness", "Conversions / Sales", "Product Demo / Reviews", "Local Reach", "Shorts Campaign"]
    )

    keywords_input = st.text_area(
        "Business Keywords (comma-separated)",
        value="kitchen gadgets, cookware, meal prep"
    )
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    st.subheader("Filters")
    min_subs = st.number_input("Min Subscribers", min_value=0, value=100000, step=10000)
    max_subs = st.number_input("Max Subscribers (0 = no limit)", min_value=0, value=0, step=10000)
    min_avg_views = st.number_input("Min Avg Views (last 10 videos)", min_value=0, value=0, step=1000)
    recency = st.selectbox("Upload Recency", ["Any", "Last 30 days", "Last 90 days"], index=2)

    run_search = st.button("🚀 Find Channels", type="primary")

# ==================================================
# Header
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write(
    "Find the best YouTube channels to invest in using **region**, **language**, **keyword relevance**, "
    "**engagement quality**, and **channel activity**."
)
st.markdown("<div class='small-note'>⚠️ If hosted on Render free tier, initial load may take 30–60 seconds (cold start).</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Run Search
# ==================================================
if run_search:
    if not keywords:
        st.warning("Please enter at least one keyword.")
        st.stop()

    query = " ".join(keywords[:5])  # keep query short for API search

    with st.spinner("Searching channels..."):
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            type="channel",
            maxResults=20,
            regionCode=region_code
        ).execute()

    channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
    if not channel_ids:
        st.warning("No channels found. Try different keywords.")
        st.stop()

    with st.spinner("Fetching channel stats..."):
        channels_response = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(channel_ids)
        ).execute()

    rows = []

    with st.spinner("Scoring channels..."):
        for ch in channels_response.get("items", []):
            snippet = ch.get("snippet", {})
            stats = ch.get("statistics", {})

            channel_id = ch.get("id", "")
            title = snippet.get("title", "")
            desc = snippet.get("description", "")

            subs = safe_int(stats.get("subscriberCount", 0))
            total_views = safe_int(stats.get("viewCount", 0))
            video_count = safe_int(stats.get("videoCount", 0))

            # Subscriber filters
            if subs < min_subs:
                continue
            if max_subs and subs > max_subs:
                continue

            # Get uploads playlist
            try:
                playlist_res = youtube.channels().list(
                    part="contentDetails",
                    id=channel_id
                ).execute()
                uploads_id = playlist_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            except Exception:
                continue

            # Get last 10 videos (snippets)
            vids_res = youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_id,
                maxResults=10
            ).execute()

            video_ids = []
            video_titles = []
            published_days = []

            for it in vids_res.get("items", []):
                sn = it.get("snippet", {})
                video_titles.append(sn.get("title", ""))
                publishedAt = sn.get("publishedAt", "")
                if publishedAt:
                    published_days.append(days_ago(publishedAt))

                vid = sn.get("resourceId", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            if not video_ids:
                continue

            # Get video view counts
            vstats = youtube.videos().list(
                part="statistics",
                id=",".join(video_ids)
            ).execute()

            views_list = [safe_int(v.get("statistics", {}).get("viewCount", 0)) for v in vstats.get("items", [])]
            avg_views = int(sum(views_list) / max(len(views_list), 1))

            if min_avg_views and avg_views < min_avg_views:
                continue

            inactive_days = min(published_days) if published_days else 9999
            uploads_90d = sum(1 for d in published_days if d <= 90)

            # Recency filter
            if recency == "Last 30 days" and inactive_days > 30:
                continue
            if recency == "Last 90 days" and inactive_days > 90:
                continue

            geo_lang_conf = infer_geo_lang_confidence(region, language, title, desc, video_titles)

            match_score, relevance, engagement, activity, geo_lang, penalty, ratio = compute_match_score(
                keywords=keywords,
                channel_text=f"{title} {desc}",
                video_titles=video_titles,
                subs=subs,
                avg_views=avg_views,
                uploads_90d=uploads_90d,
                inactive_days=inactive_days,
                geo_lang_conf=geo_lang_conf
            )

            # Badge
            if ratio >= 0.15:
                badge = "🟢 High Engagement"
            elif ratio >= 0.05:
                badge = "🟡 Average Engagement"
            else:
                badge = "🔴 Low Engagement"

            rows.append({
                "Channel": title,
                "Subscribers": subs,
                "Total Views": total_views,
                "Avg Views (Last 10)": avg_views,
                "Uploads (Last 90d)": uploads_90d,
                "Inactive Days": inactive_days,
                "Geo/Lang Confidence": geo_lang,
                "Relevance": relevance,
                "Engagement": round(engagement, 1),
                "Activity": round(activity, 1),
                "Penalty": penalty,
                "Match Score": round(match_score, 1),
                "Badge": badge,
                "Channel URL": f"https://www.youtube.com/channel/{channel_id}",
            })

    if not rows:
        st.warning("No channels matched your filters. Try lowering min subscribers or min avg views.")
        st.stop()

    df = pd.DataFrame(rows).sort_values("Match Score", ascending=False)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Ranked Channel Recommendations")

    st.dataframe(
        df[[
            "Channel", "Subscribers", "Total Views", "Avg Views (Last 10)",
            "Uploads (Last 90d)", "Geo/Lang Confidence", "Match Score", "Badge", "Channel URL"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="youtube_channel_recommendations.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("How Match Score is calculated")
    st.markdown(
        """
- **Relevance (0–40):** keyword overlap in channel description + recent video titles  
- **Engagement (0–25):** avg views per video divided by subscribers  
- **Activity (0–20):** uploads in last 90 days  
- **Geo/Lang (0–10):** simple text-based confidence score  
- **Penalty (-15..0):** inactivity and very low engagement  
"""
    )
    st.markdown('</div>', unsafe_allow_html=True)

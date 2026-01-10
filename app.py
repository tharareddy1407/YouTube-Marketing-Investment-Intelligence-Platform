import os
import base64
from pathlib import Path
from datetime import datetime, timezone
import re
import time
import urllib.parse as urlparse

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
# Background Image (Local file, Render-safe) + Light overlay
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
            background-image:
                linear-gradient(rgba(255,255,255,0.86), rgba(255,255,255,0.86)),
                url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_local_background("assets/background.png")

# ==================================================
# UI Styling
# ==================================================
st.markdown("""
<style>
.content-box {
    background: rgba(255, 255, 255, 0.96);
    padding: 1.7rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.status-box {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(15, 23, 42, 0.12);
    padding: 0.95rem 1.1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1.2rem 0;
}
.small-note {
    font-size: 0.9rem;
    color: #334155;
}
h1, h2, h3 { color: #0f172a; font-weight: 900; }
p, span, div, label { color: #1f2937; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Country -> Language options (Dynamic)
# ==================================================
COUNTRY_LANGUAGE_MAP = {
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

# YouTube "regionCode" expects ISO 3166-1 alpha-2 (US, IN, GB, etc.)
COUNTRY_REGION_CODE = {
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
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return set(w for w in text.split() if len(w) > 2)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def infer_geo_lang_confidence(country, language, channel_title, channel_desc, video_titles):
    """Simple MVP heuristic (0–10)."""
    text = " ".join([channel_title or "", channel_desc or ""] + (video_titles or [])).lower()
    score = 0

    if country.lower() in text:
        score += 4
    if country == "USA" and any(s in text for s in ["usa", "united states", "u.s.", "america"]):
        score += 4
    if country == "India" and any(s in text for s in ["india", "bharat", "indian"]):
        score += 4

    if language.lower() in text:
        score += 2

    return clamp(score, 0, 10)

def compute_fit_score(product_keywords, channel_text, video_titles, subs, avg_views, uploads_90d, inactive_days, geo_lang_conf):
    """
    0–100 score for "fit to invest"
    - Relevance (0–45)
    - Engagement (0–25)
    - Activity (0–20)
    - Geo/Lang confidence (0–10)
    - Penalty (-15..0)
    """
    kw_tokens = set()
    for kw in product_keywords:
        kw_tokens |= tokenize(kw)

    ch_tokens = tokenize(channel_text)
    vid_tokens = set()
    for t in video_titles:
        vid_tokens |= tokenize(t)

    # Relevance (0–45)
    hits_channel = len(kw_tokens & ch_tokens)
    hits_videos = len(kw_tokens & vid_tokens)
    relevance = clamp((hits_channel * 2) + (hits_videos * 1), 0, 45)

    # Engagement (0–25) using avg_views/subs ratio
    ratio = (avg_views / max(subs, 1)) if subs else 0.0
    engagement = 25 if ratio >= 0.30 else clamp((ratio / 0.30) * 25, 0, 25)

    # Activity (0–20) uploads in last 90d
    activity = clamp((uploads_90d / 12.0) * 20, 0, 20)

    # Geo/Lang (0–10)
    geo_lang = clamp(geo_lang_conf, 0, 10)

    # Penalty
    penalty = 0
    if inactive_days > 120:
        penalty -= 10
    if ratio < 0.02:
        penalty -= 5

    total = clamp(relevance + engagement + activity + geo_lang + penalty, 0, 100)
    return total, relevance, engagement, activity, geo_lang, penalty, ratio

def investment_recommendation(score: float):
    if score >= 70:
        return "✅ Recommended", "Strong fit for your product with healthy channel signals."
    if score >= 45:
        return "🟡 Maybe", "Some signals look good—review content match and engagement before investing."
    return "❌ Not Recommended", "Weak match or weak engagement/activity signals—consider other channels."

def extract_channel_id_or_handle(text: str):
    if not text:
        return None, None
    t = text.strip()
    if t.startswith("http"):
        u = urlparse.urlparse(t)
        path = u.path.strip("/")

        if path.startswith("channel/"):
            parts = path.split("/")
            return (parts[1] if len(parts) > 1 else None), None

        if path.startswith("@"):
            return None, path

        return None, None

    return None, None

def resolve_channel_id(youtube_client, channel_input: str):
    ch_id, handle = extract_channel_id_or_handle(channel_input)

    if ch_id:
        return ch_id

    if handle:
        try:
            resp = youtube_client.channels().list(part="id", forHandle=handle).execute()
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

    resp = youtube_client.search().list(
        q=channel_input,
        part="snippet",
        type="channel",
        maxResults=1
    ).execute()

    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["snippet"]["channelId"]

# ==================================================
# YouTube API Key
# ==================================================
API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ Missing YOUTUBE_API_KEY. Add it as an environment variable (local terminal or Render).")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

# ==================================================
# Header
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write("Choose a mode, enter your country/state/language + product, then get channels or evaluate one channel.")
st.markdown("<div class='small-note'>⚠️ Hosted on Render free tier. Cold start may take 30–60 seconds.</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

status_area = st.empty()

# ==================================================
# Mode Selection (MAIN UI)
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
mode = st.radio("Select Mode", ["Discover Channels", "Evaluate a Channel"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Common Inputs (show only these fields)
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.2, 1.2, 1.2])

with col1:
    country = st.selectbox("Country", list(COUNTRY_LANGUAGE_MAP.keys()), index=0)
with col2:
    state = st.text_input("State (optional)", placeholder="Ex: Texas / California / Telangana")
with col3:
    language = st.selectbox("Language", COUNTRY_LANGUAGE_MAP.get(country, ["English"]), index=0)

product_input = st.text_area(
    "Marketing Product / Keywords (comma-separated)",
    value="phone, smartphone, mobile accessories" if mode == "Evaluate a Channel" else "kitchen gadgets, cookware, meal prep"
)
product_keywords = [k.strip() for k in product_input.split(",") if k.strip()]

min_subs = st.number_input("Minimum Subscribers", min_value=0, value=100000, step=10000)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Mode-specific inputs (ONLY show what is needed)
# ==================================================
if mode == "Evaluate a Channel":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    channel_input = st.text_input(
        "Channel Name or URL",
        placeholder="Ex: Marques Brownlee OR https://www.youtube.com/@mkbhd"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Action button
# ==================================================
btn_label = "🚀 Find Channels" if mode == "Discover Channels" else "✅ Evaluate Channel"
run = st.button(btn_label, type="primary")

# ==================================================
# Fetch channel analysis helper
# ==================================================
def fetch_channel_analysis(channel_id: str):
    ch_resp = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    ).execute()

    items = ch_resp.get("items", [])
    if not items:
        return None

    ch = items[0]
    snippet = ch.get("snippet", {})
    stats = ch.get("statistics", {})
    cd = ch.get("contentDetails", {})

    title = snippet.get("title", "")
    desc = snippet.get("description", "")
    subs = safe_int(stats.get("subscriberCount", 0))
    total_views = safe_int(stats.get("viewCount", 0))
    uploads_id = cd.get("relatedPlaylists", {}).get("uploads")

    if not uploads_id:
        return None

    vids_res = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_id,
        maxResults=10
    ).execute()

    video_ids, video_titles, published_days = [], [], []
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
        return None

    vstats = youtube.videos().list(part="statistics", id=",".join(video_ids)).execute()
    views_list = [safe_int(v.get("statistics", {}).get("viewCount", 0)) for v in vstats.get("items", [])]
    avg_views = int(sum(views_list) / max(len(views_list), 1))

    inactive_days = min(published_days) if published_days else 9999
    uploads_90d = sum(1 for d in published_days if d <= 90)

    return {
        "channel_id": channel_id,
        "title": title,
        "desc": desc,
        "subs": subs,
        "total_views": total_views,
        "avg_views": avg_views,
        "inactive_days": inactive_days,
        "uploads_90d": uploads_90d,
        "video_titles": video_titles,
        "url": f"https://www.youtube.com/channel/{channel_id}",
    }

# ==================================================
# Run Logic
# ==================================================
if run:
    if not product_keywords:
        st.warning("Please enter your marketing product/keywords.")
        st.stop()

    region_code = COUNTRY_REGION_CODE.get(country, "US")

    status_area.markdown(
        "<div class='status-box'>🔎 <b>Checking with YouTube...</b> Results are on the way ✅</div>",
        unsafe_allow_html=True
    )
    progress = st.progress(0)
    progress.progress(15)
    time.sleep(0.12)

    try:
        if mode == "Discover Channels":
            # Search channels by product keywords in selected country
            query = " ".join(product_keywords[:6])

            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="channel",
                maxResults=25,
                regionCode=region_code
            ).execute()
            progress.progress(40)

            channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
            if not channel_ids:
                progress.empty()
                status_area.empty()
                st.warning("No channels found. Try broader keywords.")
                st.stop()

            channels_response = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=",".join(channel_ids)
            ).execute()
            progress.progress(60)

            rows = []
            for ch in channels_response.get("items", []):
                cid = ch.get("id", "")
                analysis = fetch_channel_analysis(cid)
                if not analysis:
                    continue

                # Filter by minimum subscribers only (as requested)
                if analysis["subs"] < min_subs:
                    continue

                geo_lang_conf = infer_geo_lang_confidence(
                    country, language, analysis["title"], analysis["desc"], analysis["video_titles"]
                )

                score, *_ = compute_fit_score(
                    product_keywords=product_keywords,
                    channel_text=f"{analysis['title']} {analysis['desc']}",
                    video_titles=analysis["video_titles"],
                    subs=analysis["subs"],
                    avg_views=analysis["avg_views"],
                    uploads_90d=analysis["uploads_90d"],
                    inactive_days=analysis["inactive_days"],
                    geo_lang_conf=geo_lang_conf
                )

                rows.append({
                    "Channel": analysis["title"],
                    "Subscribers": analysis["subs"],
                    "Total Views": analysis["total_views"],
                    "Avg Views (Last 10)": analysis["avg_views"],
                    "Fit Score": round(score, 1),
                    "Channel URL": analysis["url"],
                })

            progress.progress(85)

            if not rows:
                progress.empty()
                status_area.empty()
                st.warning("No channels matched. Try reducing Minimum Subscribers or using broader keywords.")
                st.stop()

            df = pd.DataFrame(rows)

            # Top channels based on subscribers (as requested)
            df = df.sort_values(["Subscribers", "Fit Score"], ascending=[False, False]).head(20)

            progress.progress(100)
            time.sleep(0.08)
            progress.empty()

            status_area.markdown(
                "<div class='status-box'>✅ <b>Results are ready!</b> Top channels based on subscribers are below.</div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("✅ Top Channels (by Subscribers)")

            # Show only the values user entered + results
            st.write(f"**Country:** {country}  |  **State:** {state or 'N/A'}  |  **Language:** {language}")
            st.write(f"**Marketing Product:** {', '.join(product_keywords)}  |  **Min Subscribers:** {min_subs:,}")

            st.dataframe(
                df[["Channel", "Subscribers", "Total Views", "Avg Views (Last 10)", "Fit Score", "Channel URL"]],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="youtube_channels_discover.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Evaluate a specific channel
            if not channel_input.strip():
                progress.empty()
                status_area.empty()
                st.warning("Please enter a channel name or URL.")
                st.stop()

            channel_id = resolve_channel_id(youtube, channel_input)
            if not channel_id:
                progress.empty()
                status_area.empty()
                st.error("Could not find that channel. Try another name or paste the channel URL.")
                st.stop()

            progress.progress(55)

            analysis = fetch_channel_analysis(channel_id)
            if not analysis:
                progress.empty()
                status_area.empty()
                st.error("Could not fetch channel details. Try again.")
                st.stop()

            if analysis["subs"] < min_subs:
                progress.empty()
                status_area.empty()
                st.warning(f"Channel has {analysis['subs']:,} subscribers, below your minimum {min_subs:,}.")
                st.stop()

            geo_lang_conf = infer_geo_lang_confidence(
                country, language, analysis["title"], analysis["desc"], analysis["video_titles"]
            )

            score, relevance, engagement, activity, geo_lang, penalty, ratio = compute_fit_score(
                product_keywords=product_keywords,
                channel_text=f"{analysis['title']} {analysis['desc']}",
                video_titles=analysis["video_titles"],
                subs=analysis["subs"],
                avg_views=analysis["avg_views"],
                uploads_90d=analysis["uploads_90d"],
                inactive_days=analysis["inactive_days"],
                geo_lang_conf=geo_lang_conf
            )

            decision, explanation = investment_recommendation(score)

            progress.progress(100)
            time.sleep(0.08)
            progress.empty()

            status_area.markdown(
                "<div class='status-box'>✅ <b>Channel evaluation completed.</b></div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("📌 Investment Decision")

            # Show only the values user entered + results (as requested)
            st.write(f"**Country:** {country}  |  **State:** {state or 'N/A'}  |  **Language:** {language}")
            st.write(f"**Marketing Product:** {', '.join(product_keywords)}  |  **Min Subscribers:** {min_subs:,}")

            st.markdown(f"### {decision}")
            st.write(explanation)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fit Score", f"{round(score, 1)}/100")
            c2.metric("Subscribers", f"{analysis['subs']:,}")
            c3.metric("Avg Views (Last 10)", f"{analysis['avg_views']:,}")
            c4.metric("Engagement Ratio", f"{ratio:.3f}")

            st.write("**Score breakdown**")
            st.write(f"- Relevance: {relevance}/45")
            st.write(f"- Engagement: {round(engagement, 1)}/25")
            st.write(f"- Activity: {round(activity, 1)}/20")
            st.write(f"- Geo/Lang Confidence: {geo_lang}/10")
            st.write(f"- Penalty: {penalty}")

            st.write("**Channel Link**")
            st.write(analysis["url"])
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        progress.empty()
        status_area.empty()
        st.error(f"Something went wrong while calling YouTube API: {e}")

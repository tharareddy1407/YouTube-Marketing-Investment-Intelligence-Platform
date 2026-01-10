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
                linear-gradient(
                    rgba(255, 255, 255, 0.84),
                    rgba(255, 255, 255, 0.84)
                ),
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

# Put your background image in repo at this path:
set_local_background("assets/background.png")

# ==================================================
# UI Styling (brighter + readable)
# ==================================================
st.markdown("""
<style>
.content-box {
    background: rgba(255, 255, 255, 0.96);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}
.small-note {
    font-size: 0.9rem;
    color: #334155;
}
.status-box {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(15, 23, 42, 0.12);
    padding: 0.95rem 1.1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1.2rem 0;
}
h1, h2, h3 { color: #0f172a; font-weight: 900; }
p, span, div, label { color: #1f2937; font-size: 1rem; }
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
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return set(w for w in text.split() if len(w) > 2)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def infer_geo_lang_confidence(region, language, channel_title, channel_desc, video_titles):
    """
    Simple MVP heuristic (0–10).
    """
    text = " ".join([channel_title or "", channel_desc or ""] + (video_titles or [])).lower()
    score = 0

    # Region signals
    if region.lower() in text:
        score += 4
    if region == "USA" and any(s in text for s in ["usa", "united states", "u.s.", "america"]):
        score += 4
    if region == "India" and any(s in text for s in ["india", "bharat", "indian"]):
        score += 4

    # Language signals
    if language.lower() in text:
        score += 2

    return clamp(score, 0, 10)

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

    # Engagement (0–25) avg_views/subs
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

def investment_recommendation(score: float):
    if score >= 70:
        return "✅ Recommended", "Strong fit for your product and healthy performance signals."
    if score >= 45:
        return "🟡 Maybe", "Some signals look good, but review content match and engagement before investing."
    return "❌ Not Recommended", "Low match or weak engagement/activity signals. Consider different channels."

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

    # 1) ID already present
    if ch_id:
        return ch_id

    # 2) Handle present
    if handle:
        try:
            resp = youtube_client.channels().list(part="id", forHandle=handle).execute()
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

    # 3) Search by name
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
# Sidebar (Mode + Inputs)
# ==================================================
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Mode",
        ["Discover Channels", "Evaluate a Channel"],
        index=0
    )

    st.subheader("Audience")
    region = st.selectbox("Region", list(REGION_LANGUAGE_MAP.keys()), index=0)
    region_code = REGION_CODE_MAP.get(region, "US")

    language_options = REGION_LANGUAGE_MAP.get(region, ["English"])
    language = st.selectbox("Language", language_options, index=0)

    st.subheader("Goal")
    goal = st.radio(
        "Investment Goal",
        ["Brand Awareness", "Conversions / Sales", "Product Demo / Reviews", "Local Reach", "Shorts Campaign"]
    )

    if mode == "Discover Channels":
        st.subheader("Business Keywords")
        keywords_input = st.text_area(
            "Keywords (comma-separated)",
            value="kitchen gadgets, cookware, meal prep",
            key="discover_keywords"
        )
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    else:
        st.subheader("Channel to Evaluate")
        channel_input = st.text_input(
            "Channel Name or URL",
            placeholder="Ex: Marques Brownlee or https://www.youtube.com/@mkbhd",
            key="eval_channel"
        )

        product_input = st.text_area(
            "What are you marketing? (keywords)",
            value="phone, smartphone, mobile accessories",
            key="eval_product"
        )
        product_keywords = [k.strip() for k in product_input.split(",") if k.strip()]

    st.subheader("Filters")
    min_subs = st.number_input("Min Subscribers", min_value=0, value=100000, step=10000, key="min_subs")
    max_subs = st.number_input("Max Subscribers (0 = no limit)", min_value=0, value=0, step=10000, key="max_subs")
    min_avg_views = st.number_input("Min Avg Views (last 10 videos)", min_value=0, value=0, step=1000, key="min_avg_views")
    recency = st.selectbox("Upload Recency", ["Any", "Last 30 days", "Last 90 days"], index=2, key="recency")

    run = st.button("🚀 Find Channels" if mode == "Discover Channels" else "✅ Evaluate Channel", type="primary")

# ==================================================
# Header
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write(
    "Discover high-ROI YouTube channels or evaluate if a specific channel is a good fit for your product "
    "based on **region**, **language**, **relevance**, **engagement**, and **activity**."
)
st.markdown("<div class='small-note'>⚠️ Hosted on Render free tier. Cold start may take 30–60 seconds.</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

status_area = st.empty()

# ==================================================
# Common: fetch channel + last 10 videos, compute metrics
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

    vstats = youtube.videos().list(
        part="statistics",
        id=",".join(video_ids)
    ).execute()

    views_list = [safe_int(v.get("statistics", {}).get("viewCount", 0)) for v in vstats.get("items", [])]
    avg_views = int(sum(views_list) / max(len(views_list), 1))

    inactive_days = min(published_days) if published_days else 9999
    uploads_90d = sum(1 for d in published_days if d <= 90)

    return {
        "title": title,
        "desc": desc,
        "subs": subs,
        "total_views": total_views,
        "avg_views": avg_views,
        "inactive_days": inactive_days,
        "uploads_90d": uploads_90d,
        "video_titles": video_titles,
        "url": f"https://www.youtube.com/channel/{channel_id}",
        "channel_id": channel_id,
    }

# ==================================================
# Run
# ==================================================
if run:
    progress = st.progress(0)

    if mode == "Discover Channels":
        if not keywords:
            st.warning("Please enter at least one keyword.")
            st.stop()

        status_area.markdown(
            "<div class='status-box'>🔎 <b>Checking with YouTube...</b> Results are on the way ✅</div>",
            unsafe_allow_html=True
        )
        progress.progress(10)
        time.sleep(0.12)

        query = " ".join(keywords[:5])

        try:
            # Search channels
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="channel",
                maxResults=25,
                regionCode=region_code
            ).execute()
            progress.progress(35)

            channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
            if not channel_ids:
                progress.empty()
                status_area.empty()
                st.warning("No channels found. Try different keywords.")
                st.stop()

            # Fetch channel details in batch
            channels_response = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=",".join(channel_ids)
            ).execute()
            progress.progress(55)

            rows = []

            for ch in channels_response.get("items", []):
                channel_id = ch.get("id", "")
                analysis = fetch_channel_analysis(channel_id)
                if not analysis:
                    continue

                subs = analysis["subs"]
                avg_views = analysis["avg_views"]

                # Filters
                if subs < min_subs:
                    continue
                if max_subs and subs > max_subs:
                    continue
                if min_avg_views and avg_views < min_avg_views:
                    continue
                if recency == "Last 30 days" and analysis["inactive_days"] > 30:
                    continue
                if recency == "Last 90 days" and analysis["inactive_days"] > 90:
                    continue

                geo_lang_conf = infer_geo_lang_confidence(
                    region, language, analysis["title"], analysis["desc"], analysis["video_titles"]
                )

                score, relevance, engagement, activity, geo_lang, penalty, ratio = compute_match_score(
                    keywords=keywords,
                    channel_text=f"{analysis['title']} {analysis['desc']}",
                    video_titles=analysis["video_titles"],
                    subs=subs,
                    avg_views=avg_views,
                    uploads_90d=analysis["uploads_90d"],
                    inactive_days=analysis["inactive_days"],
                    geo_lang_conf=geo_lang_conf
                )

                if ratio >= 0.15:
                    badge = "🟢 High Engagement"
                elif ratio >= 0.05:
                    badge = "🟡 Average Engagement"
                else:
                    badge = "🔴 Low Engagement"

                rows.append({
                    "Channel": analysis["title"],
                    "Subscribers": subs,
                    "Total Views": analysis["total_views"],
                    "Avg Views (Last 10)": avg_views,
                    "Uploads (Last 90d)": analysis["uploads_90d"],
                    "Geo/Lang Confidence": geo_lang,
                    "Match Score": round(score, 1),
                    "Badge": badge,
                    "Channel URL": analysis["url"],
                })

            progress.progress(90)

            if not rows:
                progress.empty()
                status_area.empty()

                st.markdown('<div class="content-box">', unsafe_allow_html=True)
                st.warning("No channels matched your filters.")
                st.write("Try these quick fixes:")
                st.write("- Lower **Min Subscribers** (example: 50k–200k)")
                st.write("- Set **Upload Recency** to **Any**")
                st.write("- Reduce **Min Avg Views**")
                st.write("- Try broader keywords (example: 'cooking', 'kitchen', 'recipes')")
                if st.button("✨ Relax Filters & Retry"):
                    st.session_state["min_subs"] = 100000
                    st.session_state["min_avg_views"] = 0
                    st.session_state["recency"] = "Any"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            df = pd.DataFrame(rows).sort_values("Match Score", ascending=False)

            progress.progress(100)
            time.sleep(0.08)
            progress.empty()

            status_area.markdown(
                "<div class='status-box'>✅ <b>Results are ready!</b> Scroll down to view recommended channels.</div>",
                unsafe_allow_html=True
            )

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

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"Something went wrong while calling YouTube API: {e}")

    # ==================================================
    # Evaluate a Channel
    # ==================================================
    else:
        if not channel_input.strip():
            st.warning("Please enter a channel name or URL.")
            st.stop()

        if not product_keywords:
            st.warning("Please enter product keywords (example: phone, smartphone).")
            st.stop()

        status_area.markdown(
            "<div class='status-box'>🔎 <b>Checking this channel with YouTube...</b> Results are on the way ✅</div>",
            unsafe_allow_html=True
        )
        progress.progress(15)
        time.sleep(0.12)

        try:
            channel_id = resolve_channel_id(youtube, channel_input)
            if not channel_id:
                progress.empty()
                status_area.empty()
                st.error("Could not find that channel. Try a different name or paste the channel URL.")
                st.stop()

            progress.progress(45)

            analysis = fetch_channel_analysis(channel_id)
            if not analysis:
                progress.empty()
                status_area.empty()
                st.error("Could not fetch channel details. Try again.")
                st.stop()

            # Apply filters
            subs = analysis["subs"]
            avg_views = analysis["avg_views"]

            if subs < min_subs:
                progress.empty()
                status_area.empty()
                st.warning(f"Channel subscribers ({subs:,}) are below your Min Subscribers filter ({min_subs:,}).")
                st.stop()

            if max_subs and subs > max_subs:
                progress.empty()
                status_area.empty()
                st.warning(f"Channel subscribers ({subs:,}) exceed your Max Subscribers filter ({max_subs:,}).")
                st.stop()

            if min_avg_views and avg_views < min_avg_views:
                progress.empty()
                status_area.empty()
                st.warning(f"Channel avg views ({avg_views:,}) are below your Min Avg Views filter ({min_avg_views:,}).")
                st.stop()

            if recency == "Last 30 days" and analysis["inactive_days"] > 30:
                progress.empty()
                status_area.empty()
                st.warning("Channel is not active in the last 30 days. Try changing Upload Recency to 'Any'.")
                st.stop()

            if recency == "Last 90 days" and analysis["inactive_days"] > 90:
                progress.empty()
                status_area.empty()
                st.warning("Channel is not active in the last 90 days. Try changing Upload Recency to 'Any'.")
                st.stop()

            geo_lang_conf = infer_geo_lang_confidence(
                region, language, analysis["title"], analysis["desc"], analysis["video_titles"]
            )

            score, relevance, engagement, activity, geo_lang, penalty, ratio = compute_match_score(
                keywords=product_keywords,
                channel_text=f"{analysis['title']} {analysis['desc']}",
                video_titles=analysis["video_titles"],
                subs=subs,
                avg_views=avg_views,
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
            st.subheader("📌 Channel Investment Decision")
            st.markdown(f"### {decision}")
            st.write(explanation)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fit Score", f"{round(score, 1)}/100")
            c2.metric("Subscribers", f"{subs:,}")
            c3.metric("Avg Views (Last 10)", f"{avg_views:,}")
            c4.metric("Engagement Ratio", f"{ratio:.3f}")

            st.write("**Score breakdown**")
            st.write(f"- Relevance: {relevance}/40")
            st.write(f"- Engagement: {round(engagement, 1)}/25")
            st.write(f"- Activity: {round(activity, 1)}/20")
            st.write(f"- Geo/Lang Confidence: {geo_lang}/10")
            st.write(f"- Penalty: {penalty}")

            st.write("**Channel Link**")
            st.write(analysis["url"])

            st.markdown('</div>', unsafe_allow_html=True)

            # Optional: suggest alternatives (top 5) using product keywords
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("✨ Similar channels you can consider")
            st.write("These are discovered using your product keywords (top matches).")

            query = " ".join(product_keywords[:5])
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="channel",
                maxResults=10,
                regionCode=region_code
            ).execute()
            alt_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
            alt_ids = [cid for cid in alt_ids if cid != channel_id]

            alt_rows = []
            if alt_ids:
                alt_resp = youtube.channels().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(alt_ids[:10])
                ).execute()

                for ch in alt_resp.get("items", []):
                    cid = ch.get("id", "")
                    a = fetch_channel_analysis(cid)
                    if not a:
                        continue

                    geo_lang_conf2 = infer_geo_lang_confidence(region, language, a["title"], a["desc"], a["video_titles"])
                    s2, *_ = compute_match_score(
                        keywords=product_keywords,
                        channel_text=f"{a['title']} {a['desc']}",
                        video_titles=a["video_titles"],
                        subs=a["subs"],
                        avg_views=a["avg_views"],
                        uploads_90d=a["uploads_90d"],
                        inactive_days=a["inactive_days"],
                        geo_lang_conf=geo_lang_conf2
                    )

                    alt_rows.append({
                        "Channel": a["title"],
                        "Subscribers": a["subs"],
                        "Avg Views (Last 10)": a["avg_views"],
                        "Match Score": round(s2, 1),
                        "Channel URL": a["url"]
                    })

            if alt_rows:
                alt_df = pd.DataFrame(alt_rows).sort_values("Match Score", ascending=False).head(5)
                st.dataframe(alt_df, use_container_width=True, hide_index=True)
            else:
                st.info("No similar channels found right now. Try different keywords.")

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"Something went wrong while calling YouTube API: {e}")

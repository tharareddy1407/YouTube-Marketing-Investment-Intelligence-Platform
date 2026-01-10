import os
import re
import math
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from datetime import datetime, timezone
import streamlit as st

st.set_page_config(
    page_title="YouTube Marketing Investment Intelligence Platform",
    layout="wide"
)


# ------------------------------
# Helpers
# ----------------------------
def tokenize(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 2)

def days_ago(iso_date: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - dt).days
    except Exception:
        return 9999

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# ----------------------------
# Scoring
# ----------------------------
def compute_scores(keywords, channel_text, video_titles, subs, avg_views, uploads_90d, geo_lang_conf, inactive_days):
    kw_tokens = set()
    for kw in keywords:
        kw_tokens |= tokenize(kw)

    ch_tokens = tokenize(channel_text)
    vid_tokens = set()
    for t in video_titles:
        vid_tokens |= tokenize(t)

    # A) Relevance (0-40)
    hits_channel = len(kw_tokens & ch_tokens)
    hits_videos = len(kw_tokens & vid_tokens)
    relevance = clamp((hits_channel * 2) + (hits_videos * 1), 0, 40)

    # B) Engagement (0-25) proxy: avg_views / subs
    ratio = (avg_views / max(subs, 1)) if subs else 0.0
    # Map ratio to score
    # 0.30+ -> 25, 0.15 -> 15, 0.05 -> 5
    if ratio >= 0.30:
        engagement = 25
    else:
        engagement = clamp((ratio / 0.30) * 25, 0, 25)

    # C) Activity (0-20)
    # 12 uploads/90d -> 20
    activity = clamp((uploads_90d / 12.0) * 20, 0, 20)

    # D) Geo/Lang confidence (0-10)
    geo_lang = clamp(geo_lang_conf, 0, 10)

    # E) Risk penalty (-15..0)
    penalty = 0
    if inactive_days > 120:
        penalty -= 10
    if ratio < 0.02:
        penalty -= 5

    total = clamp(relevance + engagement + activity + geo_lang + penalty, 0, 100)

    return {
        "relevance": relevance,
        "engagement": engagement,
        "activity": activity,
        "geo_lang": geo_lang,
        "penalty": penalty,
        "match_score": total,
        "engagement_ratio": ratio
    }

# ----------------------------
# YouTube API fetch
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_channels(api_key, query, region_code, max_results=15):
    youtube = build("youtube", "v3", developerKey=api_key)
    res = youtube.search().list(
        q=query,
        part="snippet",
        type="channel",
        maxResults=max_results,
        regionCode=region_code
    ).execute()

    channel_ids = [it["snippet"]["channelId"] for it in res.get("items", [])]
    return channel_ids

@st.cache_data(ttl=3600, show_spinner=False)
def get_channels_stats(api_key, channel_ids):
    youtube = build("youtube", "v3", developerKey=api_key)
    chunks = [channel_ids[i:i+50] for i in range(0, len(channel_ids), 50)]
    out = []
    for ch in chunks:
        res = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(ch),
            maxResults=50
        ).execute()
        out.extend(res.get("items", []))
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_videos(api_key, channel_id, max_videos=10):
    youtube = build("youtube", "v3", developerKey=api_key)

    # Get uploads playlist
    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    items = ch.get("items", [])
    if not items:
        return []
    uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    vids = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads,
        maxResults=max_videos
    ).execute()

    video_ids = []
    videos_meta = []
    for it in vids.get("items", []):
        sn = it["snippet"]
        vid = sn.get("resourceId", {}).get("videoId")
        if vid:
            video_ids.append(vid)
            videos_meta.append({
                "title": sn.get("title", ""),
                "publishedAt": sn.get("publishedAt", "")
            })

    if not video_ids:
        return []

    vstats = youtube.videos().list(
        part="statistics,snippet",
        id=",".join(video_ids)
    ).execute()

    stats_map = {}
    for v in vstats.get("items", []):
        stats_map[v["id"]] = {
            "views": safe_int(v.get("statistics", {}).get("viewCount", 0)),
            "title": v.get("snippet", {}).get("title", ""),
            "publishedAt": v.get("snippet", {}).get("publishedAt", "")
        }

    # Keep in original order
    out = []
    for vid in video_ids:
        out.append(stats_map.get(vid, {"views": 0, "title": "", "publishedAt": ""}))
    return out

def infer_geo_lang_confidence(region, language, channel_title, channel_desc, video_titles):
    # Very simple heuristic for MVP (0-10)
    text = " ".join([channel_title or "", channel_desc or ""] + (video_titles or [])).lower()

    score = 0
    # region signals
    if region.lower() in text:
        score += 4
    if region.lower() in ["usa", "united states", "us"] and any(s in text for s in ["usa", "united states", "us", "america"]):
        score += 4

    # language signals
    if language.lower() in text:
        score += 2
    # cap
    return clamp(score, 0, 10)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="YouTube Channel Investment Finder", layout="wide")
st.title("📺 YouTube Channel Investment Finder")
st.caption("Find the best YouTube channels to invest in based on niche relevance, engagement, activity, region & language.")

api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
if not api_key:
    st.error("Missing API key. Set environment variable YOUTUBE_API_KEY and rerun.")
    st.stop()

with st.sidebar:
    st.header("Inputs")

    region = st.selectbox("Region", ["USA", "Canada", "India", "UK", "Australia"], index=0)
    region_code_map = {"USA": "US", "Canada": "CA", "India": "IN", "UK": "GB", "Australia": "AU"}
    region_code = region_code_map.get(region, "US")

    language = st.selectbox("Language", ["English", "Spanish", "Hindi", "French"], index=0)

    goal = st.radio(
        "Investment Goal",
        ["Brand Awareness", "Conversions / Sales", "Product Demo / Reviews", "Local Reach", "Shorts Campaign"]
    )

    keywords_str = st.text_area("Business Keywords (comma-separated)", "kitchen gadgets, cookware, meal prep")
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    st.subheader("Filters")
    min_subs = st.number_input("Min Subscribers", min_value=0, value=100000, step=10000)
    max_subs = st.number_input("Max Subscribers (optional)", min_value=0, value=0, step=10000)
    min_avg_views = st.number_input("Min Avg Views (last 10 videos)", min_value=0, value=0, step=1000)
    recency = st.selectbox("Upload Recency", ["Any", "Last 30 days", "Last 90 days"], index=2)

    run = st.button("🔍 Find Channels", type="primary")

if run:
    if not keywords:
        st.warning("Please enter at least 1 keyword.")
        st.stop()

    query = " ".join(keywords[:5])  # keep query short
    with st.spinner("Searching channels..."):
        channel_ids = search_channels(api_key, query, region_code, max_results=20)

    if not channel_ids:
        st.warning("No channels found. Try different keywords.")
        st.stop()

    with st.spinner("Fetching channel stats..."):
        channels = get_channels_stats(api_key, channel_ids)

    rows = []
    with st.spinner("Fetching recent videos + scoring..."):
        for ch in channels:
            sn = ch.get("snippet", {})
            stt = ch.get("statistics", {})

            title = sn.get("title", "")
            desc = sn.get("description", "")
            channel_id = ch.get("id", "")

            subs = safe_int(stt.get("subscriberCount", 0))
            total_views = safe_int(stt.get("viewCount", 0))
            video_count = safe_int(stt.get("videoCount", 0))

            # Filters by subs
            if subs < min_subs:
                continue
            if max_subs and subs > max_subs:
                continue

            recent = get_recent_videos(api_key, channel_id, max_videos=10)
            video_titles = [v.get("title", "") for v in recent]
            views_list = [safe_int(v.get("views", 0)) for v in recent]
            avg_views = int(sum(views_list) / max(len(views_list), 1))

            # Filter by avg views
            if min_avg_views and avg_views < min_avg_views:
                continue

            # Activity metrics
            published_days = [days_ago(v.get("publishedAt", "")) for v in recent if v.get("publishedAt")]
            inactive_days = min(published_days) if published_days else 9999

            uploads_90d = sum(1 for d in published_days if d <= 90)

            # Recency filter
            if recency == "Last 30 days" and (inactive_days > 30):
                continue
            if recency == "Last 90 days" and (inactive_days > 90):
                continue

            geo_lang_conf = infer_geo_lang_confidence(region, language, title, desc, video_titles)

            scores = compute_scores(
                keywords=keywords,
                channel_text=f"{title} {desc}",
                video_titles=video_titles,
                subs=subs,
                avg_views=avg_views,
                uploads_90d=uploads_90d,
                geo_lang_conf=geo_lang_conf,
                inactive_days=inactive_days
            )

            # Badges
            ratio = scores["engagement_ratio"]
            if ratio >= 0.15:
                badge = "🟢 High Engagement"
            elif ratio >= 0.05:
                badge = "🟡 Average Engagement"
            else:
                badge = "🔴 Low Engagement"

            rows.append({
                "Channel": title,
                "Channel ID": channel_id,
                "Subscribers": subs,
                "Total Views": total_views,
                "Avg Views (last 10)": avg_views,
                "Uploads (last 90d)": uploads_90d,
                "Inactive Days": inactive_days,
                "Geo/Lang Confidence": scores["geo_lang"],
                "Relevance": scores["relevance"],
                "Engagement": scores["engagement"],
                "Activity": scores["activity"],
                "Penalty": scores["penalty"],
                "Match Score": scores["match_score"],
                "Badge": badge,
                "URL": f"https://www.youtube.com/channel/{channel_id}"
            })

    if not rows:
        st.warning("No channels matched your filters. Try lowering min subs or min avg views.")
        st.stop()

    df = pd.DataFrame(rows).sort_values("Match Score", ascending=False)

    st.subheader("✅ Ranked Channel Recommendations")
    st.dataframe(
        df[["Channel", "Subscribers", "Total Views", "Avg Views (last 10)", "Uploads (last 90d)", "Geo/Lang Confidence", "Match Score", "Badge", "URL"]],
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
    st.subheader("How scores are calculated (MVP)")
    st.write("""
- **Relevance (0–40):** keyword hits in channel text + recent video titles  
- **Engagement (0–25):** avg views / subscribers  
- **Activity (0–20):** upload count in last 90 days  
- **Geo/Lang (0–10):** basic inference from text  
- **Penalty (-15..0):** inactivity + very low engagement  
""")

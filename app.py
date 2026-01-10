import os
import base64
from pathlib import Path
from datetime import datetime, timezone
import re

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build

# --------------------------------------------------
# Page config (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="YouTube Marketing Investment Intelligence Platform",
    layout="wide"
)

# --------------------------------------------------
# Background image loader (LOCAL FILE)
# --------------------------------------------------
def set_local_background(image_path: str):
    img_path = Path(image_path)
    if not img_path.exists():
        st.error(f"Background image not found: {image_path}")
        return

    encoded = base64.b64encode(img_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 CHANGE PATH IF NEEDED
set_local_background("assets/background.png")

# --------------------------------------------------
# UI Styling
# --------------------------------------------------
st.markdown("""
<style>
.content-box {
    background: rgba(255, 255, 255, 0.90);
    padding: 2rem;
    border-radius: 14px;
    margin-bottom: 2rem;
}
.small-note {
    font-size: 0.85rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def tokenize(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return set(w for w in text.split() if len(w) > 2)

def days_ago(iso_date: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 9999

def safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

# --------------------------------------------------
# YouTube API setup
# --------------------------------------------------
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    st.error("❌ YouTube API key not found. Please set YOUTUBE_API_KEY as an environment variable.")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
with st.sidebar:
    st.header("🔍 Campaign Inputs")

    region = st.selectbox("Region", ["USA", "Canada", "UK", "India", "Australia"])
    region_map = {"USA": "US", "Canada": "CA", "UK": "GB", "India": "IN", "Australia": "AU"}
    region_code = region_map[region]

    language = st.selectbox("Language", ["English", "Spanish", "Hindi", "French"])

    goal = st.radio(
        "Investment Goal",
        ["Brand Awareness", "Conversions", "Product Demo", "Local Reach", "Shorts Campaign"]
    )

    keywords_input = st.text_area(
        "Business Keywords",
        placeholder="kitchen gadgets, cookware, meal prep"
    )
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    st.subheader("Filters")
    min_subs = st.number_input("Minimum Subscribers", value=100000, step=10000)
    min_avg_views = st.number_input("Minimum Avg Views (last 10 videos)", value=0, step=1000)

    run_search = st.button("🚀 Find Channels")

# --------------------------------------------------
# Main Content
# --------------------------------------------------
st.markdown('<div class="content-box">', unsafe_allow_html=True)

st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write(
    "Identify **high-ROI YouTube channels** for marketing investment based on "
    "region, language, niche relevance, engagement quality, and activity."
)

st.markdown(
    "<p class='small-note'>⚠️ App is hosted on Render free tier. "
    "Cold start may take 30–60 seconds.</p>",
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Run search
# --------------------------------------------------
if run_search:
    if not keywords:
        st.warning("Please enter at least one keyword.")
        st.stop()

    with st.spinner("Searching YouTube channels..."):
        search_response = youtube.search().list(
            q=" ".join(keywords),
            part="snippet",
            type="channel",
            maxResults=20,
            regionCode=region_code
        ).execute()

    channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]

    if not channel_ids:
        st.warning("No channels found. Try different keywords.")
        st.stop()

    channels_response = youtube.channels().list(
        part="snippet,statistics",
        id=",".join(channel_ids)
    ).execute()

    results = []

    for ch in channels_response.get("items", []):
        stats = ch["statistics"]
        snippet = ch["snippet"]

        subs = safe_int(stats.get("subscriberCount"))
        if subs < min_subs:
            continue

        channel_id = ch["id"]

        playlist_res = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()

        uploads_id = playlist_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        videos_res = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_id,
            maxResults=10
        ).execute()

        video_titles = []
        video_ids = []

        for v in videos_res.get("items", []):
            video_titles.append(v["snippet"]["title"])
            video_ids.append(v["snippet"]["resourceId"]["videoId"])

        if not video_ids:
            continue

        video_stats = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        ).execute()

        views = [safe_int(v["statistics"].get("viewCount")) for v in video_stats.get("items", [])]
        avg_views = sum(views) // max(len(views), 1)

        if avg_views < min_avg_views:
            continue

        engagement_ratio = round(avg_views / max(subs, 1), 3)

        results.append({
            "Channel": snippet["title"],
            "Subscribers": subs,
            "Avg Views (Last 10)": avg_views,
            "Engagement Ratio": engagement_ratio,
            "Channel URL": f"https://www.youtube.com/channel/{channel_id}"
        })

    if not results:
        st.warning("No channels matched your filters.")
        st.stop()

    df = pd.DataFrame(results).sort_values("Engagement Ratio", ascending=False)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Recommended Channels")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        file_name="youtube_channel_recommendations.csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)

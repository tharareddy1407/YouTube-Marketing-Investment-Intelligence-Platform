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
        return

    encoded = base64.b64encode(img_path.read_bytes()).decode()
    ext = img_path.suffix.lower().replace(".", "") or "png"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.88)),
                url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .content-box {{
            background: rgba(255, 255, 255, 0.96);
            padding: 1.7rem;
            border-radius: 16px;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.10);
        }}

        .status-box {{
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(15, 23, 42, 0.12);
            padding: 0.95rem 1.1rem;
            border-radius: 14px;
            margin: 0.8rem 0 1.2rem 0;
        }}

        .small-note {{
            font-size: 0.9rem;
            color: #334155;
        }}

        h1, h2, h3 {{ color: #0f172a; font-weight: 900; }}
        p, span, div, label {{ color: #1f2937; font-size: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_local_background("assets/background.png")

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
    kw_tokens = set()
    for kw in product_keywords:
        kw_tokens |= tokenize(kw)

    ch_tokens = tokenize(channel_text)
    vid_tokens = set()
    for t in video_titles:
        vid_tokens |= tokenize(t)

    relevance = clamp((len(kw_tokens & ch_tokens) * 2) + (len(kw_tokens & vid_tokens) * 1), 0, 45)

    ratio = (avg_views / max(subs, 1)) if subs else 0.0
    engagement = 25 if ratio >= 0.30 else clamp((ratio / 0.30) * 25, 0, 25)

    activity = clamp((uploads_90d / 12.0) * 20, 0, 20)
    geo_lang = clamp(geo_lang_conf, 0, 10)

    penalty = 0
    if inactive_days > 120:
        penalty -= 10
    if ratio < 0.02:
        penalty -= 5

    total = clamp(relevance + engagement + activity + geo_lang + penalty, 0, 100)
    return total, relevance, engagement, activity, geo_lang, penalty, ratio

def investment_recommendation(score: float):
    if score >= 70:
        return "✅ Recommended", "Strong fit and healthy channel signals."
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
# Session State
# ==================================================
if "mode" not in st.session_state:
    st.session_state["mode"] = None  # None -> landing page

# ==================================================
# Header
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write("Select one option to continue.")
st.markdown("<div class='small-note'>⚠️ Hosted on Render free tier. Cold start may take 30–60 seconds.</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

status_area = st.empty()

# ==================================================
# Landing Page: Two options in the middle
# ==================================================
if st.session_state["mode"] is None:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 1])

    with mid:
        st.subheader("Choose Mode")
        if st.button("🔎 Discover Channels", use_container_width=True):
            st.session_state["mode"] = "discover"
            st.rerun()

        if st.button("✅ Evaluate a Channel", use_container_width=True):
            st.session_state["mode"] = "evaluate"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==================================================
# Mode: Discover Channels (only these inputs)
# ==================================================
if st.session_state["mode"] == "discover":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🔎 Discover Channels")

    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", list(COUNTRY_LANGUAGE_MAP.keys()), index=0)
    with c2:
        state = st.text_input("State", placeholder="Ex: Texas / California / Telangana")
    with c3:
        language = st.selectbox("Language", COUNTRY_LANGUAGE_MAP.get(country, ["English"]), index=0)

    product_input = st.text_input("Marketing Product", placeholder="Ex: phone, kitchen gadgets, skincare")
    min_subs = st.number_input("Minimum Subscribers", min_value=0, value=100000, step=10000)

    run = st.button("🚀 Find Channels", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if not product_input.strip():
            st.warning("Please enter a marketing product.")
            st.stop()

        region_code = COUNTRY_REGION_CODE.get(country, "US")
        product_keywords = [product_input.strip()]

        status_area.markdown(
            "<div class='status-box'>🔎 <b>Checking with YouTube...</b> Results are on the way ✅</div>",
            unsafe_allow_html=True
        )

        progress = st.progress(0)
        progress.progress(15)
        time.sleep(0.12)

        try:
            query = product_input.strip()

            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="channel",
                maxResults=25,
                regionCode=region_code
            ).execute()
            progress.progress(45)

            channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
            if not channel_ids:
                progress.empty()
                status_area.empty()
                st.warning("No channels found. Try a broader product keyword.")
                st.stop()

            channels_response = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=",".join(channel_ids)
            ).execute()
            progress.progress(65)

            rows = []
            for ch in channels_response.get("items", []):
                cid = ch.get("id", "")
                analysis = fetch_channel_analysis(cid)
                if not analysis:
                    continue

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

            progress.progress(90)

            if not rows:
                progress.empty()
                status_area.empty()
                st.warning("No channels matched. Try reducing Minimum Subscribers or changing product keyword.")
                st.stop()

            df = pd.DataFrame(rows).sort_values("Subscribers", ascending=False).head(20)

            progress.progress(100)
            time.sleep(0.08)
            progress.empty()

            status_area.markdown(
                "<div class='status-box'>✅ <b>Results are ready!</b> Top channels by subscribers are below.</div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("✅ Results")
            st.write(f"**Country:** {country} | **State:** {state} | **Language:** {language}")
            st.write(f"**Marketing Product:** {product_input} | **Minimum Subscribers:** {min_subs:,}")

            st.dataframe(
                df[["Channel", "Subscribers", "Total Views", "Avg Views (Last 10)", "Fit Score", "Channel URL"]],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="discover_channels.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"Something went wrong while calling YouTube API: {e}")

# ==================================================
# Mode: Evaluate a Channel (only this input)
# ==================================================
if st.session_state["mode"] == "evaluate":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Evaluate a Channel")
    channel_input = st.text_input(
        "Channel Name or URL",
        placeholder="Ex: Marques Brownlee OR https://www.youtube.com/@mkbhd"
    )
    run = st.button("✅ Evaluate Channel", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if not channel_input.strip():
            st.warning("Please enter a channel name or URL.")
            st.stop()

        status_area.markdown(
            "<div class='status-box'>🔎 <b>Checking this channel with YouTube...</b> Results are on the way ✅</div>",
            unsafe_allow_html=True
        )

        progress = st.progress(0)
        progress.progress(20)
        time.sleep(0.12)

        try:
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

            # Simple evaluation: Just show channel stats
            progress.progress(100)
            time.sleep(0.08)
            progress.empty()

            status_area.markdown(
                "<div class='status-box'>✅ <b>Channel details are ready!</b></div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("📌 Channel Result")
            st.write(f"**Channel:** {analysis['title']}")
            st.write(f"**Subscribers:** {analysis['subs']:,}")
            st.write(f"**Total Views:** {analysis['total_views']:,}")
            st.write(f"**Avg Views (Last 10):** {analysis['avg_views']:,}")
            st.write(f"**Channel URL:** {analysis['url']}")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"Something went wrong while calling YouTube API: {e}")

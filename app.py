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

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="YouTube Marketing Investment Intelligence Platform",
    layout="wide",
)

# =========================
# CSS: remove blocks + style cards
# =========================
st.markdown("""
<style>
/* Hide Streamlit chrome */
header[data-testid="stHeader"]{display:none!important;}
div[data-testid="stToolbar"]{display:none!important;}
div[data-testid="stDecoration"]{display:none!important;}
footer{visibility:hidden!important;}

/* Remove the gray/white “pill blocks” (Streamlit layout wrappers) */
div[data-testid="stVerticalBlock"] { background: transparent !important; }
div[data-testid="stVerticalBlock"] > div { background: transparent !important; box-shadow: none !important; }
section.main > div { background: transparent !important; }

/* Reduce top padding */
.block-container { padding-top: 0.8rem !important; }

/* Cards */
.content-box{
  background: rgba(255,255,255,0.965);
  padding: 1.8rem;
  border-radius: 18px;
  margin-bottom: 1.25rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.status-box{
  background: rgba(255,255,255,0.94);
  border: 1px solid rgba(15,23,42,0.12);
  padding: 0.95rem 1.1rem;
  border-radius: 14px;
  margin: 0.8rem 0 1.2rem 0;
}
.small-note{font-size:0.92rem;color:#334155;}
h1,h2,h3{color:#0f172a;font-weight:900;}
p,span,div,label{color:#1f2937;font-size:1rem;}
.stButton>button{border-radius:14px!important;padding:0.8rem 1rem!important;font-weight:750!important;}
</style>
""", unsafe_allow_html=True)

# =========================
# Background image
# =========================
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
        </style>
        """,
        unsafe_allow_html=True
    )

set_local_background("assets/background.png")

# =========================
# Country -> language + region code
# =========================
COUNTRY_LANGUAGE_MAP = {
    "USA": ["English", "Spanish"],
    "India": ["Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Urdu", "Odia"],
    "UK": ["English", "Welsh", "Scottish Gaelic", "Irish"],
    "Canada": ["English", "French"],
    "Australia": ["English"],
    "UAE": ["Arabic", "English", "Hindi", "Urdu"],
    "Singapore": ["English", "Mandarin", "Malay", "Tamil"],
}
COUNTRY_REGION_CODE = {"USA":"US","India":"IN","UK":"GB","Canada":"CA","Australia":"AU","UAE":"AE","Singapore":"SG"}

# =========================
# Helpers
# =========================
def safe_int(val, default=0):
    try: return int(val)
    except Exception: return default

def days_ago(iso_date: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z","+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 9999

def engagement_label(r: float) -> str:
    if r >= 0.15: return "🟢 High"
    if r >= 0.05: return "🟡 Average"
    return "🔴 Low"

def infer_channel_type(title: str, desc: str, recent_titles: list[str]) -> str:
    text = " ".join([title or "", desc or ""] + (recent_titles or [])).lower()
    taxonomy = [
        ("Tech & Gadgets", ["tech","iphone","android","smartphone","mobile","laptop","review","unboxing","gadget"]),
        ("Cooking & Food", ["recipe","cooking","kitchen","chef","baking","food","meal prep","dosa","biryani"]),
        ("Beauty & Fashion", ["makeup","skincare","beauty","fashion","outfit","haul"]),
        ("Fitness & Health", ["fitness","workout","gym","yoga","health","diet"]),
        ("Education", ["tutorial","learn","course","lecture","explained","how to","tips"]),
        ("Finance & Business", ["finance","stock","invest","trading","business","marketing","money"]),
        ("Entertainment", ["comedy","movie","cinema","music","funny","prank"]),
        ("Travel", ["travel","vlog","trip","tour","hotel"]),
        ("Gaming", ["gaming","gameplay","walkthrough","ps5","xbox","minecraft","fortnite"]),
        ("News & Politics", ["news","politics","breaking","debate"]),
    ]
    best, best_score = "General", 0
    for label, kws in taxonomy:
        score = sum(1 for k in kws if k in text)
        if score > best_score:
            best, best_score = label, score
    return best

def extract_channel_id_or_handle(text: str):
    if not text: return None, None
    t = text.strip()
    if t.startswith("http"):
        u = urlparse.urlparse(t)
        path = u.path.strip("/")
        if path.startswith("channel/"):
            parts = path.split("/")
            return (parts[1] if len(parts)>1 else None), None
        if path.startswith("@"):
            return None, path
    return None, None

def resolve_channel_id(youtube_client, channel_input: str):
    ch_id, handle = extract_channel_id_or_handle(channel_input)
    if ch_id: return ch_id
    if handle:
        try:
            resp = youtube_client.channels().list(part="id", forHandle=handle).execute()
            items = resp.get("items", [])
            if items: return items[0]["id"]
        except Exception:
            pass
    resp = youtube_client.search().list(q=channel_input, part="snippet", type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items: return None
    return items[0]["snippet"]["channelId"]

# =========================
# YouTube API
# =========================
API_KEY = os.getenv("YOUTUBE_API_KEY","").strip()
if not API_KEY:
    st.error("❌ Missing YOUTUBE_API_KEY. Add it in Render → Environment Variables.")
    st.stop()
youtube = build("youtube","v3", developerKey=API_KEY)

def fetch_channel_analysis(channel_id: str):
    ch_resp = youtube.channels().list(part="snippet,statistics,contentDetails", id=channel_id).execute()
    items = ch_resp.get("items", [])
    if not items: return None
    ch = items[0]
    snippet = ch.get("snippet", {})
    stats = ch.get("statistics", {})
    cd = ch.get("contentDetails", {})
    title = snippet.get("title","")
    desc = snippet.get("description","")
    subs = safe_int(stats.get("subscriberCount",0))
    total_views = safe_int(stats.get("viewCount",0))
    video_count = safe_int(stats.get("videoCount",0))
    uploads_id = cd.get("relatedPlaylists", {}).get("uploads")
    if not uploads_id: return None

    vids_res = youtube.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=10).execute()
    video_ids, video_titles, published_days = [], [], []
    for it in vids_res.get("items", []):
        sn = it.get("snippet", {})
        video_titles.append(sn.get("title",""))
        publishedAt = sn.get("publishedAt","")
        if publishedAt:
            published_days.append(days_ago(publishedAt))
        vid = sn.get("resourceId", {}).get("videoId")
        if vid: video_ids.append(vid)
    if not video_ids: return None

    vstats = youtube.videos().list(part="statistics", id=",".join(video_ids)).execute()
    views_list = [safe_int(v.get("statistics", {}).get("viewCount",0)) for v in vstats.get("items", [])]
    avg_views = int(sum(views_list) / max(len(views_list), 1))
    inactive_days = min(published_days) if published_days else 9999
    uploads_90d = sum(1 for d in published_days if d <= 90)

    engagement_ratio = avg_views / max(subs, 1)
    return {
        "channel_id": channel_id,
        "title": title,
        "subs": subs,
        "total_views": total_views,
        "video_count": video_count,
        "avg_views": avg_views,
        "inactive_days": inactive_days,
        "uploads_90d": uploads_90d,
        "video_titles": video_titles,
        "engagement_ratio": float(engagement_ratio),
        "engagement_label": engagement_label(float(engagement_ratio)),
        "channel_type": infer_channel_type(title, desc, video_titles),
        "url": f"https://www.youtube.com/channel/{channel_id}",
    }

# =========================
# State
# =========================
if "mode" not in st.session_state:
    st.session_state["mode"] = None

# =========================
# Header card
# =========================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write("Choose a mode below to continue.")
st.markdown("<div class='small-note'>⚠️ Hosted on Render free tier. Cold start may take 30–60 seconds.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

status_area = st.empty()

# =========================
# Landing
# =========================
if st.session_state["mode"] is None:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.subheader("Choose Mode")
        if st.button("🔎 Discover Channels", use_container_width=True):
            st.session_state["mode"] = "discover"
            st.rerun()
        if st.button("✅ Evaluate a Channel", use_container_width=True):
            st.session_state["mode"] = "evaluate"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# Discover
# =========================
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

    product = st.text_input("Marketing Product", placeholder="Ex: phone, kitchen gadgets, skincare")
    min_subs = st.number_input("Minimum Subscribers", min_value=0, value=100000, step=10000)
    run = st.button("🚀 Find Channels", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not product.strip():
            st.warning("Please enter a marketing product.")
            st.stop()

        region_code = COUNTRY_REGION_CODE.get(country, "US")
        status_area.markdown("<div class='status-box'>🔎 <b>Checking with YouTube...</b> Results are on the way ✅</div>", unsafe_allow_html=True)

        progress = st.progress(0)
        progress.progress(25)
        time.sleep(0.08)

        search_response = youtube.search().list(
            q=product.strip(),
            part="snippet",
            type="channel",
            maxResults=25,
            regionCode=region_code
        ).execute()
        progress.progress(60)

        channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
        if not channel_ids:
            progress.empty(); status_area.empty()
            st.warning("No channels found. Try a broader keyword.")
            st.stop()

        rows = []
        for cid in channel_ids:
            a = fetch_channel_analysis(cid)
            if not a:
                continue
            if a["subs"] < min_subs:
                continue
            rows.append({
                "Channel": a["title"],
                "Type": a["channel_type"],
                "Subscribers": a["subs"],
                "Avg Views (Last 10)": a["avg_views"],
                "Engagement": f"{a['engagement_label']} ({a['engagement_ratio']:.3f})",
                "Total Views": a["total_views"],
                "Channel URL": a["url"],
            })

        if not rows:
            progress.empty(); status_area.empty()
            st.warning("No channels matched. Try reducing Minimum Subscribers.")
            st.stop()

        df = pd.DataFrame(rows).sort_values("Subscribers", ascending=False).head(20)

        progress.progress(100); time.sleep(0.06); progress.empty()
        status_area.markdown("<div class='status-box'>✅ <b>Results are ready!</b></div>", unsafe_allow_html=True)

        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("✅ Results")
        st.write(f"**Country:** {country} | **State:** {state or 'N/A'} | **Language:** {language}")
        st.write(f"**Marketing Product:** {product} | **Minimum Subscribers:** {min_subs:,}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "discover_channels.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Evaluate
# =========================
if st.session_state["mode"] == "evaluate":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Evaluate a Channel")
    channel_input = st.text_input("Channel Name or URL", placeholder="Ex: Marques Brownlee or https://www.youtube.com/@mkbhd")
    run = st.button("✅ Evaluate Channel", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not channel_input.strip():
            st.warning("Please enter a channel name or URL.")
            st.stop()

        status_area.markdown("<div class='status-box'>🔎 <b>Checking this channel with YouTube...</b> Results are on the way ✅</div>", unsafe_allow_html=True)
        progress = st.progress(0)
        progress.progress(30)
        time.sleep(0.08)

        channel_id = resolve_channel_id(youtube, channel_input)
        if not channel_id:
            progress.empty(); status_area.empty()
            st.error("Could not find that channel. Try another name or paste the channel URL.")
            st.stop()

        a = fetch_channel_analysis(channel_id)
        if not a:
            progress.empty(); status_area.empty()
            st.error("Could not fetch channel details. Try again.")
            st.stop()

        progress.progress(100); time.sleep(0.06); progress.empty()
        status_area.markdown("<div class='status-box'>✅ <b>Channel evaluation completed.</b></div>", unsafe_allow_html=True)

        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📌 Channel Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subscribers", f"{a['subs']:,}")
        c2.metric("Avg Views (Last 10)", f"{a['avg_views']:,}")
        c3.metric("Engagement Score", f"{a['engagement_ratio']:.3f}")
        c4.metric("Engagement Level", a["engagement_label"])
        st.write(f"**Channel Name:** {a['title']}")
        st.write(f"**Channel Type:** {a['channel_type']}")
        st.write(f"**Total Views:** {a['total_views']:,}")
        st.write(f"**Total Videos:** {a['video_count']:,}")
        st.write(f"**Uploads (Last 90 days):** {a['uploads_90d']}")
        st.write(f"**Last Upload (days ago):** {a['inactive_days']}")
        st.write("**Channel URL:**")
        st.write(a["url"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📺 Recent Video Titles (Last 10)")
        for t in a["video_titles"]:
            if t.strip():
                st.write(f"• {t}")
        st.markdown("</div>", unsafe_allow_html=True)

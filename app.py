import os
import re
import time
import base64
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse as urlparse
from typing import Dict, List

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build

# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="YouTube Marketing Investment Intelligence Platform",
    layout="wide",
)

# ==================================================
# Background image (local -> base64)
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
        </style>
        """,
        unsafe_allow_html=True,
    )

set_local_background("assets/background.png")

# ==================================================
# Country -> languages + YouTube region code
# ==================================================
COUNTRY_LANGUAGE_MAP: Dict[str, List[str]] = {
    "USA": ["English", "Spanish"],
    "India": ["Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Urdu", "Odia"],
    "UK": ["English", "Welsh", "Scottish Gaelic", "Irish"],
    "Canada": ["English", "French"],
    "Australia": ["English"],
    "UAE": ["Arabic", "English", "Hindi", "Urdu"],
    "Singapore": ["English", "Mandarin", "Malay", "Tamil"],
}
COUNTRY_REGION_CODE = {"USA": "US", "India": "IN", "UK": "GB", "Canada": "CA", "Australia": "AU", "UAE": "AE", "Singapore": "SG"}

LANG_TO_YT_CODE = {
    "English": "en", "Spanish": "es", "French": "fr",
    "Hindi": "hi", "Telugu": "te", "Tamil": "ta", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa", "Urdu": "ur", "Odia": "or",
    "Arabic": "ar", "Mandarin": "zh", "Malay": "ms",
}

LANG_UNICODE_RANGES = {
    "Hindi": r"[\u0900-\u097F]",
    "Marathi": r"[\u0900-\u097F]",
    "Telugu": r"[\u0C00-\u0C7F]",
    "Tamil": r"[\u0B80-\u0BFF]",
    "Kannada": r"[\u0C80-\u0CFF]",
    "Malayalam": r"[\u0D00-\u0D7F]",
    "Bengali": r"[\u0980-\u09FF]",
    "Gujarati": r"[\u0A80-\u0AFF]",
    "Punjabi": r"[\u0A00-\u0A7F]",
    "Odia": r"[\u0B00-\u0B7F]",
    "Urdu": r"[\u0600-\u06FF]",
    "Arabic": r"[\u0600-\u06FF]",
}

LANG_QUERY_HINTS = {
    "Spanish": " español en español",
    "French": " français en français",
    "Hindi": " हिंदी Hindi",
    "Telugu": " తెలుగు Telugu",
    "Tamil": " தமிழ் Tamil",
    "Kannada": " ಕನ್ನಡ Kannada",
    "Malayalam": " മലയാളം Malayalam",
    "Marathi": " मराठी Marathi",
    "Bengali": " বাংলা Bengali",
    "Gujarati": " ગુજરાતી Gujarati",
    "Punjabi": " ਪੰਜਾਬੀ Punjabi",
    "Urdu": " اردو Urdu",
    "Odia": " ଓଡ଼ିଆ Odia",
    "Arabic": " العربية Arabic",
    "Mandarin": " 中文 Mandarin",
    "Malay": " Bahasa Melayu Malay",
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

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if len(t) >= 3]

def engagement_label(r: float) -> str:
    if r >= 0.15:
        return "🟢 High"
    if r >= 0.05:
        return "🟡 Average"
    return "🔴 Low"

def channel_matches_language(video_titles: List[str], language: str) -> bool:
    titles_text = " ".join(video_titles or [])
    pattern = LANG_UNICODE_RANGES.get(language)
    if pattern:
        return bool(re.search(pattern, titles_text))

    t = titles_text.lower()

    if language == "Spanish":
        return (any(w in t for w in [" el ", " la ", " de ", " y ", " que ", " para ", " con "])
                or any(c in t for c in "áéíóúñ"))

    if language == "French":
        return (any(w in t for w in [" le ", " la ", " de ", " et ", " pour ", " avec ", " que "])
                or any(c in t for c in "àâçéèêëîïôùûüÿœ"))

    if language == "English":
        return not bool(re.search(
            r"[\u0600-\u06FF\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0CFF\u0D00-\u0D7F]",
            titles_text
        ))

    return True

def language_purity_percent(video_titles: List[str], language: str) -> int:
    if not video_titles:
        return 0
    hits = sum(1 for t in video_titles if channel_matches_language([t], language))
    return int(round((hits / len(video_titles)) * 100))

def infer_channel_type(title: str, desc: str, recent_titles: List[str]) -> str:
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

    resp = youtube_client.search().list(q=channel_input, part="snippet", type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["snippet"]["channelId"]

# ==================================================
# YouTube API setup
# ==================================================
API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ Missing YOUTUBE_API_KEY. Add it in Render → Environment Variables.")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_channel_analysis(channel_id: str):
    """
    Returns: channel stats + last 10 titles + aligned last 10 view counts
    """
    ch_resp = youtube.channels().list(part="snippet,statistics,contentDetails", id=channel_id).execute()
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
    video_count = safe_int(stats.get("videoCount", 0))

    uploads_id = cd.get("relatedPlaylists", {}).get("uploads")
    if not uploads_id:
        return None

    vids_res = youtube.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=10).execute()

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
    views_map = {v["id"]: safe_int(v.get("statistics", {}).get("viewCount", 0)) for v in vstats.get("items", [])}
    views_list = [views_map.get(vid, 0) for vid in video_ids]

    avg_views = int(sum(views_list) / max(len(views_list), 1))
    inactive_days = min(published_days) if published_days else 9999
    uploads_90d = sum(1 for d in published_days if d <= 90)
    engagement_ratio = avg_views / max(subs, 1)

    return {
        "channel_id": channel_id,
        "title": title,
        "desc": desc,
        "subs": subs,
        "total_views": total_views,
        "video_count": video_count,
        "avg_views": avg_views,
        "inactive_days": inactive_days,
        "uploads_90d": uploads_90d,
        "video_titles": video_titles,
        "views_list": views_list,
        "engagement_ratio": float(engagement_ratio),
        "engagement_label": engagement_label(float(engagement_ratio)),
        "channel_type": infer_channel_type(title, desc, video_titles),
        "url": f"https://www.youtube.com/channel/{channel_id}",
    }

# ==================================================
# 9 Insights calculations
# ==================================================
SPONSOR_WORDS = ["sponsored", "ad", "paid partnership", "promo", "promotion", "brought to you by", "partnered with"]

def calc_fit_score(product: str, title: str, desc: str, video_titles: List[str]) -> int:
    p_tokens = set(tokenize(product))
    if not p_tokens:
        return 0
    c_tokens = set(tokenize(title + " " + desc + " " + " ".join(video_titles)))
    overlap = len(p_tokens & c_tokens)
    return int(round(min(1.0, overlap / max(3, len(p_tokens))) * 100))

def calc_sponsorship_readiness(eng_ratio: float, uploads_90d: int, inactive_days: int) -> str:
    score = 0
    if eng_ratio >= 0.05: score += 1
    if eng_ratio >= 0.10: score += 1
    if uploads_90d >= 6: score += 1
    if uploads_90d >= 10: score += 1
    if inactive_days <= 30: score += 1
    if inactive_days <= 14: score += 1
    if score >= 5: return "🟢 High"
    if score >= 3: return "🟡 Medium"
    return "🔴 Low"

def calc_growth_momentum(views_list: List[int]) -> str:
    if not views_list or len(views_list) < 6:
        return "N/A"
    last3 = views_list[:3]
    prev = views_list[3:]
    a = sum(last3) / max(1, len(last3))
    b = sum(prev) / max(1, len(prev))
    if b <= 0:
        return "N/A"
    pct = ((a - b) / b) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"

def calc_sponsor_saturation(video_titles: List[str]) -> str:
    t = " ".join(video_titles or []).lower()
    matches = sum(1 for w in SPONSOR_WORDS if w in t)
    if matches >= 3: return "🔴 High"
    if matches >= 1: return "🟡 Medium"
    return "🟢 Low"

def calc_audience_trust(eng_ratio: float, views_list: List[int]) -> str:
    if not views_list:
        return "N/A"
    mean = sum(views_list) / max(1, len(views_list))
    if mean <= 0:
        return "N/A"
    var = sum((v - mean) ** 2 for v in views_list) / max(1, len(views_list))
    std = var ** 0.5
    cv = std / mean
    score = 0
    if eng_ratio >= 0.05: score += 1
    if eng_ratio >= 0.10: score += 1
    if cv <= 1.2: score += 1
    if cv <= 0.8: score += 1
    if score >= 4: return "🟢 Strong"
    if score >= 2: return "🟡 Medium"
    return "🔴 Weak"

def calc_product_compat(channel_type: str, product: str) -> str:
    p = normalize_text(product)
    if channel_type == "Tech & Gadgets" and any(k in p for k in ["phone","iphone","android","laptop","gadget","camera","headphone"]):
        return "🟢 Excellent"
    if channel_type == "Cooking & Food" and any(k in p for k in ["kitchen","cook","cookware","pan","recipe","spice","mixer"]):
        return "🟢 Excellent"
    if channel_type == "Beauty & Fashion" and any(k in p for k in ["skincare","makeup","serum","fashion","outfit"]):
        return "🟢 Excellent"
    return "🟡 Mixed"

def calc_risk_flags(eng_ratio: float, inactive_days: int, uploads_90d: int, views_list: List[int]) -> str:
    flags = []
    if inactive_days > 90: flags.append("Inactive 90+ days")
    if uploads_90d < 2: flags.append("Low posting")
    if eng_ratio < 0.02: flags.append("Low engagement")
    if views_list:
        mean = sum(views_list) / max(1, len(views_list))
        if mean > 0:
            var = sum((v - mean) ** 2 for v in views_list) / max(1, len(views_list))
            cv = (var ** 0.5) / mean
            if cv > 1.5:
                flags.append("High volatility")
    return "✅ None" if not flags else "⚠️ " + "; ".join(flags)

# ==================================================
# Session state: mode
# ==================================================
if "mode" not in st.session_state:
    st.session_state["mode"] = None

# ==================================================
# Header UI
# ==================================================
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📺 YouTube Marketing Investment Intelligence Platform")
st.write("Choose a mode below to continue.")
st.markdown("<div class='small-note'>⚠️ Render free tier may take 30–60 seconds on first load.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

status_area = st.empty()

# Back to Home (top-left), only when not landing
if st.session_state.get("mode") is not None:
    col_btn, _ = st.columns([2, 10])
    with col_btn:
        if st.button("⬅️ Back to Home"):
            st.session_state["mode"] = None
            st.rerun()

# ==================================================
# Landing
# ==================================================
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

# ==================================================
# DISCOVER CHANNELS
# - 4 separate tables (base + tier1 + tier2 + tier3)
# ==================================================
if st.session_state["mode"] == "discover":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🔎 Discover Channels")

    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox("Country", list(COUNTRY_LANGUAGE_MAP.keys()), index=0)
    with c2:
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
        relevance_lang = LANG_TO_YT_CODE.get(language, "en")
        lang_hint = LANG_QUERY_HINTS.get(language, "")
        query = f"{product.strip()} {lang_hint}".strip()

        status_area.markdown("<div class='status-box'>🔎 <b>Checking with YouTube...</b> Results are on the way ✅</div>", unsafe_allow_html=True)
        progress = st.progress(0)
        progress.progress(20)
        time.sleep(0.08)

        try:
            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="channel",
                maxResults=25,
                regionCode=region_code,
                relevanceLanguage=relevance_lang
            ).execute()

            progress.progress(55)

            channel_ids = [item["snippet"]["channelId"] for item in search_response.get("items", [])]
            if not channel_ids:
                progress.empty()
                status_area.empty()
                st.warning("No channels found. Try a broader keyword.")
                st.stop()

            rows = []
            for cid in channel_ids:
                a = fetch_channel_analysis(cid)
                if not a:
                    continue

                if a["subs"] < min_subs:
                    continue

                # Selected language only
                if not channel_matches_language(a["video_titles"], language):
                    continue

                eff_raw = a["avg_views"] / max(1.0, (a["subs"] / 1000.0))

                rows.append({
                    "Channel": a["title"],
                    "Type": a["channel_type"],
                    "Subscribers": a["subs"],
                    "Avg Views": a["avg_views"],
                    "Engagement": f"{a['engagement_label']} ({a['engagement_ratio']:.3f})",
                    "Channel URL": a["url"],

                    "Fit Score": calc_fit_score(product, a["title"], a["desc"], a["video_titles"]),
                    "Sponsorship Readiness": calc_sponsorship_readiness(a["engagement_ratio"], a["uploads_90d"], a["inactive_days"]),
                    "_eff_raw": eff_raw,

                    "Growth Momentum": calc_growth_momentum(a["views_list"]),
                    "Sponsor Saturation": calc_sponsor_saturation(a["video_titles"]),
                    "Audience Trust": calc_audience_trust(a["engagement_ratio"], a["views_list"]),

                    "Language Purity": f"{language_purity_percent(a['video_titles'], language)}%",
                    "Product Compatibility": calc_product_compat(a["channel_type"], product),
                    "Risk Flags": calc_risk_flags(a["engagement_ratio"], a["inactive_days"], a["uploads_90d"], a["views_list"]),
                })

            if not rows:
                progress.empty()
                status_area.empty()
                st.warning("No channels matched your filters + selected language. Try lowering subscribers or changing product keyword.")
                st.stop()

            df = pd.DataFrame(rows)

            # Normalize Cost Efficiency into “x” vs median
            med = df["_eff_raw"].median() if df["_eff_raw"].notna().any() else 0
            df["Cost Efficiency"] = (df["_eff_raw"] / med).map(lambda x: f"{x:.1f}x") if med and med > 0 else "N/A"
            df.drop(columns=["_eff_raw"], inplace=True, errors="ignore")

            df = df.sort_values(["Subscribers"], ascending=[False]).head(20)

            progress.progress(100)
            time.sleep(0.06)
            progress.empty()
            status_area.markdown("<div class='status-box'>✅ <b>Results are ready!</b></div>", unsafe_allow_html=True)

            # TABLE 1 (Base)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 1 — Base Channel Metrics")
            base_cols = ["Channel", "Type", "Subscribers", "Avg Views", "Engagement", "Channel URL"]
            st.dataframe(df[base_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 2 (Tier 1)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 2 — Tier 1")
            t1_cols = ["Channel", "Type", "Fit Score", "Sponsorship Readiness", "Cost Efficiency"]
            st.dataframe(df[t1_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 3 (Tier 2)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 3 — Tier 2")
            t2_cols = ["Channel", "Type", "Growth Momentum", "Sponsor Saturation", "Audience Trust"]
            st.dataframe(df[t2_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 4 (Tier 3)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 4 — Tier 3")
            t3_cols = ["Channel", "Type", "Language Purity", "Product Compatibility", "Risk Flags"]
            st.dataframe(df[t3_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"YouTube API error: {e}")

# ==================================================
# EVALUATE A CHANNEL
# - Add Table 1, Table 2 (Tier1), Table 3 (Tier2), Table 4 (Tier3)
# ==================================================
if st.session_state["mode"] == "evaluate":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Evaluate a Channel")

    # We need country+language here for Tier-3 language purity. Keep minimal but required.
    c1, c2 = st.columns(2)
    with c1:
        country_eval = st.selectbox("Country (for language scoring)", list(COUNTRY_LANGUAGE_MAP.keys()), index=0, key="eval_country")
    with c2:
        language_eval = st.selectbox("Language (for language scoring)", COUNTRY_LANGUAGE_MAP.get(country_eval, ["English"]), index=0, key="eval_language")

    channel_input = st.text_input("Channel Name or URL", placeholder="Ex: Marques Brownlee or https://www.youtube.com/@mkbhd")
    product_eval = st.text_input("Marketing Product (for fit scoring)", placeholder="Ex: phone, kitchen gadgets, skincare")

    run_eval = st.button("✅ Evaluate Channel", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_eval:
        if not channel_input.strip():
            st.warning("Please enter a channel name or URL.")
            st.stop()
        if not product_eval.strip():
            st.warning("Please enter your marketing product (needed for Fit Score / Compatibility).")
            st.stop()

        status_area.markdown("<div class='status-box'>🔎 <b>Checking this channel with YouTube...</b> Results are on the way ✅</div>", unsafe_allow_html=True)
        progress = st.progress(0)
        progress.progress(30)
        time.sleep(0.08)

        try:
            channel_id = resolve_channel_id(youtube, channel_input)
            if not channel_id:
                progress.empty()
                status_area.empty()
                st.error("Could not find that channel. Try another name or paste the channel URL.")
                st.stop()

            a = fetch_channel_analysis(channel_id)
            if not a:
                progress.empty()
                status_area.empty()
                st.error("Could not fetch channel details. Try again.")
                st.stop()

            # Compute all Tier insights for this single channel
            eff_raw = a["avg_views"] / max(1.0, (a["subs"] / 1000.0))
            cost_eff = "1.0x"  # single-channel baseline

            row = {
                "Channel": a["title"],
                "Type": a["channel_type"],
                "Subscribers": a["subs"],
                "Avg Views": a["avg_views"],
                "Engagement": f"{a['engagement_label']} ({a['engagement_ratio']:.3f})",
                "Channel URL": a["url"],

                "Fit Score": calc_fit_score(product_eval, a["title"], a["desc"], a["video_titles"]),
                "Sponsorship Readiness": calc_sponsorship_readiness(a["engagement_ratio"], a["uploads_90d"], a["inactive_days"]),
                "Cost Efficiency": cost_eff,

                "Growth Momentum": calc_growth_momentum(a["views_list"]),
                "Sponsor Saturation": calc_sponsor_saturation(a["video_titles"]),
                "Audience Trust": calc_audience_trust(a["engagement_ratio"], a["views_list"]),

                "Language Purity": f"{language_purity_percent(a['video_titles'], language_eval)}%",
                "Product Compatibility": calc_product_compat(a["channel_type"], product_eval),
                "Risk Flags": calc_risk_flags(a["engagement_ratio"], a["inactive_days"], a["uploads_90d"], a["views_list"]),
            }

            df_eval = pd.DataFrame([row])

            progress.progress(100)
            time.sleep(0.06)
            progress.empty()
            status_area.markdown("<div class='status-box'>✅ <b>Channel evaluation completed.</b></div>", unsafe_allow_html=True)

            # TABLE 1 (Base)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 1 — Base Channel Metrics")
            base_cols = ["Channel", "Type", "Subscribers", "Avg Views", "Engagement", "Channel URL"]
            st.dataframe(df_eval[base_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 2 (Tier 1)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 2 — Tier 1")
            t1_cols = ["Channel", "Type", "Fit Score", "Sponsorship Readiness", "Cost Efficiency"]
            st.dataframe(df_eval[t1_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 3 (Tier 2)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 3 — Tier 2")
            t2_cols = ["Channel", "Type", "Growth Momentum", "Sponsor Saturation", "Audience Trust"]
            st.dataframe(df_eval[t2_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 4 (Tier 3)
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 4 — Tier 3")
            t3_cols = ["Channel", "Type", "Language Purity", "Product Compatibility", "Risk Flags"]
            st.dataframe(df_eval[t3_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Optional: show recent titles
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("📺 Recent Video Titles (Last 10)")
            for t in a["video_titles"]:
                if t.strip():
                    st.write(f"• {t}")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            progress.empty()
            status_area.empty()
            st.error(f"YouTube API error: {e}")

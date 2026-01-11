import os
import re
import time
import base64
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse as urlparse
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build

# ==================================================
# Page config (MUST be first Streamlit call)
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
# Country -> Languages + Region code
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

# Unicode script ranges (for strict language filtering in Discover mode)
LANG_UNICODE_RANGES = {
    "Hindi": r"[\u0900-\u097F]",        # Devanagari
    "Marathi": r"[\u0900-\u097F]",      # Devanagari
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
# Expanded Channel Type Keywords (for product typing + optional improvements)
# ==================================================
TYPE_KEYWORDS = {
    "Tech & Gadgets": [
        "tech","technology","gadgets","gadget","electronics","electronic","device","devices",
        "review","unboxing","comparison","vs","hands on","hands-on","first look","benchmarks",
        "smartphone","phone","mobile","iphone","android","samsung","pixel","oneplus","xiaomi","realme",
        "laptop","macbook","windows","chromebook","tablet","ipad","surface",
        "smartwatch","smart watch","watch","wearable","fitness band","band","tracker",
        "earbuds","earphones","headphones","tws","bluetooth","wireless",
        "camera","dslr","mirrorless","lens","gimbal","microphone","mic",
        "charger","adapter","power bank","cable","usb","type c","type-c","usb-c",
        "router","wifi","wi-fi","network","internet","isp",
        "pc build","gpu","cpu","ram","ssd","hard disk","motherboard",
        "ai","artificial intelligence","apps","software","update","ios","android update",
    ],
    "Cooking & Food": [
        "food","cooking","cook","recipe","recipes","kitchen","chef","home chef",
        "cake","cakes","baking","bake","bakery","dessert","desserts","sweet","sweets",
        "bread","cookies","brownie","cupcake","pastry","ice cream","chocolate",
        "breakfast","lunch","dinner","meal","meal prep","meal-prep",
        "snack","snacks","street food","fast food","restaurant","hotel","cafe","buffet",
        "biryani","dosa","idli","vada","sambar","chutney","curry","dal","roti","paratha",
        "veg","vegetarian","non veg","non-veg","vegan","grill","tandoor","bbq","fry",
        "spices","masala","pickle","pickles","sauce","ketchup","mayonnaise",
        "cookware","pan","pot","pressure cooker","air fryer","mixer","grinder","oven",
    ],
    "Beauty & Fashion": [
        "beauty","makeup","skincare","skin care","cosmetics","fashion","style","styling",
        "outfit","outfits","haul","try on","try-on","lookbook","ootd",
        "dress","saree","lehenga","kurta","jeans","shoes","heels","sneakers",
        "accessories","jewelry","jewellery",
        "lipstick","foundation","concealer","eyeliner","mascara","blush","highlighter",
        "hair","haircare","hair care","hairstyle","salon","grooming",
        "perfume","fragrance","deodorant",
    ],
    "Fitness & Health": [
        "fitness","workout","gym","exercise","training","yoga","pilates","cardio",
        "diet","nutrition","healthy","health","wellness","weight loss","fat loss",
        "muscle","strength","bodybuilding","protein","supplement","vitamins",
        "meditation","mental health","mindfulness","stretch","mobility",
    ],
    "Education & Tutorials": [
        "education","tutorial","how to","how-to","learn","course","lesson","classes","training",
        "explained","guide","tips","tricks","basics","beginner","advanced",
        "study","exam","school","college","university",
        "coding","programming","python","java","sql","excel","power bi","tableau",
        "data","analytics","machine learning","ml","ai tutorial",
    ],
    "Finance & Business": [
        "finance","business","money","investment","investing","stocks","share market","trading",
        "crypto","bitcoin","mutual fund","bank","loan","credit","tax","budget",
        "startup","entrepreneur","marketing","digital marketing","ads","seo",
        "profit","revenue","income","side hustle","passive income",
    ],
    "Entertainment": [
        "entertainment","comedy","funny","skit","prank","reaction","roast",
        "movie","cinema","film","trailer","music","song","dance",
        "celebrity","interview","show","standup","stand-up",
    ],
    "Gaming": [
        "gaming","gameplay","walkthrough","live","stream","streaming",
        "ps5","playstation","xbox","pc gaming","mobile gaming",
        "pubg","bgmi","free fire","fortnite","minecraft","valorant","gta","cod","call of duty",
        "esports","e-sports",
    ],
    "Travel & Lifestyle": [
        "travel","trip","tour","vlog","journey","vacation","holiday",
        "hotel","resort","flight","airport","visa",
        "road trip","roadtrip","budget travel","luxury travel",
        "lifestyle","daily vlog","routine","day in my life","minimalism",
    ],
    "News & Politics": [
        "news","politics","current affairs","breaking","debate","analysis",
        "election","government","policy","law","parliament","court",
    ],
    "Kids & Family": [
        "kids","children","baby","cartoon","nursery","rhymes","toy","toys",
        "abc","alphabets","numbers","family","parenting",
    ],
    "Auto & Vehicles": [
        "car","cars","bike","bikes","automobile","vehicle","engine",
        "review","test drive","mileage","ev","electric vehicle",
        "modification","accessories","tyres","tires",
    ],
    "Sports": [
        "sports","cricket","football","soccer","tennis","badminton","kabaddi",
        "match","highlights","analysis","score","ipl","world cup",
    ],
    "Science & Knowledge": [
        "science","space","nasa","isro","physics","chemistry","biology",
        "facts","research","experiment","innovation","future","ai","technology explained",
    ],
}

# product synonym boost (helps “watch” map to tech, “cake” to food)
PRODUCT_SYNONYMS = {
    "watch": ["watch", "smartwatch", "smart watch", "wearable", "fitness band", "band", "tracker"],
    "cake": ["cake", "cakes", "cupcake", "cupcakes", "pastry", "dessert", "baking"],
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

    # Strict script match where possible
    if pattern:
        return bool(re.search(pattern, titles_text))

    # Latin-script languages: best-effort heuristics
    t = f" {titles_text.lower()} "
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
    """
    Your existing heuristic classifier.
    (You can later replace with TYPE_KEYWORDS-weighted scoring if you want.)
    """
    text = " ".join([title or "", desc or ""] + (recent_titles or [])).lower()
    taxonomy = [
        ("Tech & Gadgets", ["tech","iphone","android","smartphone","mobile","laptop","review","unboxing","gadget","watch","smartwatch"]),
        ("Cooking & Food", ["recipe","cooking","kitchen","chef","baking","food","meal prep","dosa","biryani","cake"]),
        ("Beauty & Fashion", ["makeup","skincare","beauty","fashion","outfit","haul"]),
        ("Fitness & Health", ["fitness","workout","gym","yoga","health","diet"]),
        ("Education & Tutorials", ["tutorial","learn","course","lecture","explained","how to","tips"]),
        ("Finance & Business", ["finance","stock","invest","trading","business","marketing","money"]),
        ("Entertainment", ["comedy","movie","cinema","music","funny","prank","reaction"]),
        ("Travel & Lifestyle", ["travel","vlog","trip","tour","hotel","lifestyle","routine"]),
        ("Gaming", ["gaming","gameplay","walkthrough","ps5","xbox","minecraft","fortnite","bgmi"]),
        ("News & Politics", ["news","politics","breaking","debate"]),
        ("Auto & Vehicles", ["car","bike","ev","test drive","automobile"]),
        ("Sports", ["cricket","football","ipl","match","highlights"]),
        ("Kids & Family", ["kids","rhymes","cartoon","toys"]),
        ("Science & Knowledge", ["science","space","nasa","isro","physics","facts"]),
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
# Product typing + verdict (watch->tech, cake->food, etc.)
# ==================================================
def detect_type_from_text(text: str) -> str:
    """
    Detect best matching type based on keyword hits.
    Returns one of TYPE_KEYWORDS keys or 'General'.
    """
    t = (text or "").lower().strip()
    if not t:
        return "General"

    # synonym boost
    for base, syns in PRODUCT_SYNONYMS.items():
        if any(s in t for s in syns):
            t += f" {base}"

    best_type = "General"
    best_score = 0
    for type_name, kws in TYPE_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in t:
                score += 1
        if score > best_score:
            best_score = score
            best_type = type_name

    return best_type if best_score > 0 else "General"

def investment_verdict_simple(channel_type: str, product_text: str, fit_score: int, engagement_ratio: float) -> Tuple[str, str]:
    """
    Returns (verdict_label, reason)
    verdict_label: '🟢 Recommended' | '🟡 Maybe' | '🔴 Not Recommended'
    """
    product_type = detect_type_from_text(product_text)
    p = (product_text or "").lower()

    # Category match => never harsh reject just because fit overlap is low
    if product_type != "General" and channel_type == product_type:
        if fit_score >= 55 or engagement_ratio >= 0.05:
            return "🟢 Recommended", f"Product type **{product_type}** matches channel type **{channel_type}**."
        return "🟡 Maybe", "Type matches, but keyword Fit Score is low. Try a small pilot campaign."

    # If product type is unclear
    if product_type == "General":
        if fit_score >= 60:
            return "🟡 Maybe", "Product type is unclear, but Fit Score is decent. Use a more specific keyword to improve accuracy."
        return "🔴 Not Recommended", "Product keyword is too generic and Fit Score is low. Use a more specific product name."

    # Explicit mismatch examples (cake on tech => reject, but kitchen gadgets on tech => maybe)
    if channel_type == "Tech & Gadgets" and product_type == "Cooking & Food":
        kitchen_gadget_terms = ["air fryer","mixer","grinder","oven","cookware","pan","pot","kitchen gadget","kitchen","blender"]
        if any(k in p for k in kitchen_gadget_terms):
            return "🟡 Maybe", "Tech channels can work for **kitchen gadgets** (review/unboxing angle). Consider a test sponsorship."
        return "🔴 Not Recommended", "Food items like **cake** usually don’t convert well on tech-focused audiences."

    # Default mismatch
    if fit_score >= 70 and engagement_ratio >= 0.05:
        return "🟡 Maybe", f"Different categories (**{product_type}** vs **{channel_type}**) but strong Fit Score + engagement suggests it may still work."
    return "🔴 Not Recommended", f"Mismatch: product type **{product_type}** vs channel type **{channel_type}**."

# ==================================================
# Evaluate-mode language purity (auto-detect dominant script)
# ==================================================
SCRIPT_BUCKETS: List[Tuple[str, str]] = [
    ("Telugu", r"[\u0C00-\u0C7F]"),
    ("Devanagari", r"[\u0900-\u097F]"),
    ("Tamil", r"[\u0B80-\u0BFF]"),
    ("Kannada", r"[\u0C80-\u0CFF]"),
    ("Malayalam", r"[\u0D00-\u0D7F]"),
    ("Bengali", r"[\u0980-\u09FF]"),
    ("Gujarati", r"[\u0A80-\u0AFF]"),
    ("Gurmukhi", r"[\u0A00-\u0A7F]"),
    ("Odia", r"[\u0B00-\u0B7F]"),
    ("ArabicScript", r"[\u0600-\u06FF]"),
]

def detect_dominant_script_purity(video_titles: List[str]) -> str:
    if not video_titles:
        return "N/A"
    total = len(video_titles)
    counts = {name: 0 for name, _ in SCRIPT_BUCKETS}
    latin_count = 0

    for t in video_titles:
        matched_any = False
        for name, pattern in SCRIPT_BUCKETS:
            if re.search(pattern, t or ""):
                counts[name] += 1
                matched_any = True
        if not matched_any:
            latin_count += 1

    best_name = "Latin"
    best_count = latin_count
    for name, c in counts.items():
        if c > best_count:
            best_name = name
            best_count = c

    purity = int(round((best_count / total) * 100))
    return f"{purity}% ({best_name})"

# ==================================================
# YouTube API
# ==================================================
API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ Missing YOUTUBE_API_KEY. Add it in Render → Environment Variables.")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_channel_analysis(channel_id: str):
    """
    Returns channel stats + last 10 titles + aligned view/like/comment counts.
    Like/comment counts may be missing -> 0.
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

    views_map, likes_map, comments_map = {}, {}, {}
    for v in vstats.get("items", []):
        vid = v.get("id")
        s = v.get("statistics", {}) or {}
        views_map[vid] = safe_int(s.get("viewCount", 0))
        likes_map[vid] = safe_int(s.get("likeCount", 0))
        comments_map[vid] = safe_int(s.get("commentCount", 0))

    views_list = [views_map.get(vid, 0) for vid in video_ids]
    likes_list = [likes_map.get(vid, 0) for vid in video_ids]
    comments_list = [comments_map.get(vid, 0) for vid in video_ids]

    avg_views = int(sum(views_list) / max(len(views_list), 1))
    avg_likes = int(sum(likes_list) / max(len(likes_list), 1))
    avg_comments = int(sum(comments_list) / max(len(comments_list), 1))

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
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
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
# Insights (Tier 1/2/3)
# ==================================================
SPONSOR_WORDS = ["sponsored", "ad", "paid partnership", "promo", "promotion", "brought to you by", "partnered with"]

def calc_fit_score(product: str, title: str, desc: str, video_titles: List[str]) -> int:
    # overlap-based fit score 0-100
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
    # compare last3 vs previous videos
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
    # proxy: engagement + view stability (lower volatility better)
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
    # simple compatibility: use detected product type vs channel type
    product_type = detect_type_from_text(product)
    if product_type != "General" and product_type == channel_type:
        return "🟢 Excellent"
    if channel_type == "Tech & Gadgets" and product_type == "Cooking & Food":
        # allow kitchen gadgets
        p = (product or "").lower()
        kitchen_gadget_terms = ["air fryer","mixer","grinder","oven","cookware","pan","pot","kitchen gadget","kitchen","blender"]
        if any(k in p for k in kitchen_gadget_terms):
            return "🟡 Mixed"
        return "🔴 Poor"
    if product_type == "General":
        return "🟡 Mixed"
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
# Glossary
# ==================================================
def render_glossary():
    with st.expander("ℹ️ Metrics Glossary (What each insight means)"):
        st.markdown("""
**Table 1 — Base**
- **Channel**: YouTube channel name.
- **Type**: Estimated category based on recent content.
- **Subscribers**: Total channel subscribers.
- **Avg Views**: Average views on the last 10 uploads.
- **Avg Likes / Avg Comments**: Average likes/comments on the last 10 uploads (**may be 0 if creator hides them**).
- **Engagement**: Avg Views ÷ Subscribers (proxy for audience activity).
- **Channel URL**: Direct link to the channel.

**Table 2 — Tier 1**
- **Fit Score**: 0–100 keyword match between product and channel content (titles/description).
- **Sponsorship Readiness**: High/Medium/Low based on engagement + posting frequency + recency.
- **Cost Efficiency**: ROI proxy.  
  - Discover: normalized vs median (e.g., 1.3x).  
  - Evaluate: views per 1k subs.

**Table 3 — Tier 2**
- **Growth Momentum**: last 3 videos vs older recent videos (positive % = improving reach).
- **Sponsor Saturation**: how often sponsor/ad terms appear (Low/Medium/High).
- **Audience Trust**: engagement + stability proxy (Strong/Medium/Weak).

**Table 4 — Tier 3**
- **Language Purity**:  
  - Discover: % of recent titles matching selected language.  
  - Evaluate: dominant script purity (auto-detected).
- **Product Compatibility**: how naturally your product fits the channel type.
- **Risk Flags**: inactivity, low posting, low engagement, or high volatility.

**Investment Verdict (Evaluate mode)**
- Final recommendation based on product type vs channel type + Fit Score + engagement.
""")

# ==================================================
# Session state
# ==================================================
if "mode" not in st.session_state:
    st.session_state["mode"] = None

# ==================================================
# Header
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
# DISCOVER MODE
# Tables:
# 1) Channel, Type, Subscribers, Avg Views, Avg Likes, Avg Comments, Engagement, URL
# 2) Channel, Type, Fit Score, Sponsorship Readiness, Cost Efficiency
# 3) Channel, Type, Growth Momentum, Sponsor Saturation, Audience Trust
# 4) Channel, Type, Language Purity, Product Compatibility, Risk Flags
# ==================================================
if st.session_state["mode"] == "discover":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🔎 Discover Channels")

    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox("Country", list(COUNTRY_LANGUAGE_MAP.keys()), index=0)
    with c2:
        language = st.selectbox("Language", COUNTRY_LANGUAGE_MAP.get(country, ["English"]), index=0)

    product = st.text_input("Marketing Product", placeholder="Ex: watch, phone, cake, air fryer, skincare")
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

                # Enforce selected-language channels only
                if not channel_matches_language(a["video_titles"], language):
                    continue

                fit = calc_fit_score(product, a["title"], a["desc"], a["video_titles"])
                sponsor_ready = calc_sponsorship_readiness(a["engagement_ratio"], a["uploads_90d"], a["inactive_days"])
                growth = calc_growth_momentum(a["views_list"])
                sat = calc_sponsor_saturation(a["video_titles"])
                trust = calc_audience_trust(a["engagement_ratio"], a["views_list"])
                purity = f"{language_purity_percent(a['video_titles'], language)}%"
                compat = calc_product_compat(a["channel_type"], product)
                risk = calc_risk_flags(a["engagement_ratio"], a["inactive_days"], a["uploads_90d"], a["views_list"])

                eff_raw = a["avg_views"] / max(1.0, (a["subs"] / 1000.0))  # views per 1k subs (raw for normalization)

                rows.append({
                    # Table 1
                    "Channel": a["title"],
                    "Type": a["channel_type"],
                    "Subscribers": a["subs"],
                    "Avg Views": a["avg_views"],
                    "Avg Likes": a["avg_likes"],
                    "Avg Comments": a["avg_comments"],
                    "Engagement": f"{a['engagement_label']} ({a['engagement_ratio']:.3f})",
                    "Channel URL": a["url"],

                    # Table 2 (Tier 1)
                    "Fit Score": fit,
                    "Sponsorship Readiness": sponsor_ready,
                    "_eff_raw": eff_raw,

                    # Table 3 (Tier 2)
                    "Growth Momentum": growth,
                    "Sponsor Saturation": sat,
                    "Audience Trust": trust,

                    # Table 4 (Tier 3)
                    "Language Purity": purity,
                    "Product Compatibility": compat,
                    "Risk Flags": risk,
                })

            if not rows:
                progress.empty()
                status_area.empty()
                st.warning("No channels matched your filters + selected language. Try lowering subscribers or changing product keyword.")
                st.stop()

            df = pd.DataFrame(rows)

            # Normalize cost efficiency as x vs median
            med = df["_eff_raw"].median() if df["_eff_raw"].notna().any() else 0
            df["Cost Efficiency"] = (df["_eff_raw"] / med).map(lambda x: f"{x:.1f}x") if med and med > 0 else "N/A"
            df.drop(columns=["_eff_raw"], inplace=True, errors="ignore")

            df = df.sort_values(["Subscribers"], ascending=[False]).head(20)

            progress.progress(100)
            time.sleep(0.06)
            progress.empty()
            status_area.markdown("<div class='status-box'>✅ <b>Results are ready!</b></div>", unsafe_allow_html=True)

            render_glossary()

            # TABLE 1
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 1 — Base Channel Metrics")
            base_cols = ["Channel", "Type", "Subscribers", "Avg Views", "Avg Likes", "Avg Comments", "Engagement", "Channel URL"]
            st.dataframe(df[base_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 2
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 2 — Tier 1")
            t1_cols = ["Channel", "Type", "Fit Score", "Sponsorship Readiness", "Cost Efficiency"]
            st.dataframe(df[t1_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 3
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 3 — Tier 2")
            t2_cols = ["Channel", "Type", "Growth Momentum", "Sponsor Saturation", "Audience Trust"]
            st.dataframe(df[t2_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 4
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
# EVALUATE MODE
# - No country/language input
# - Adds Investment Verdict (watch tech => recommended, cake tech => not recommended)
# - Table 1 includes Avg Likes + Avg Comments
# ==================================================
if st.session_state["mode"] == "evaluate":
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("✅ Evaluate a Channel")

    channel_input = st.text_input("Channel Name or URL", placeholder="Ex: https://www.youtube.com/@Tharareddy")
    product_eval = st.text_input("Marketing Product (for Fit Score & Compatibility)", placeholder="Ex: watch, iphone, cake, air fryer")

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

            fit = calc_fit_score(product_eval, a["title"], a["desc"], a["video_titles"])
            verdict_label, verdict_reason = investment_verdict_simple(
                channel_type=a["channel_type"],
                product_text=product_eval,
                fit_score=fit,
                engagement_ratio=a["engagement_ratio"]
            )

            # Evaluate cost efficiency as raw (views per 1k subs)
            eff_raw = a["avg_views"] / max(1.0, (a["subs"] / 1000.0))
            cost_eff = f"{eff_raw:.1f} views/1k subs"

            detected_purity = detect_dominant_script_purity(a["video_titles"])

            progress.progress(100)
            time.sleep(0.06)
            progress.empty()
            status_area.markdown("<div class='status-box'>✅ <b>Channel evaluation completed.</b></div>", unsafe_allow_html=True)

            # Verdict card
            st.markdown(
                f"""
                <div class="verdict-box">
                  <h3>Investment Verdict: {verdict_label}</h3>
                  <div class="small-note">{verdict_reason}</div>
                  <div class="small-note"><b>Detected Product Type:</b> {detect_type_from_text(product_eval)} &nbsp; | &nbsp; <b>Channel Type:</b> {a["channel_type"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_glossary()

            row = {
                # Table 1
                "Channel": a["title"],
                "Type": a["channel_type"],
                "Subscribers": a["subs"],
                "Avg Views": a["avg_views"],
                "Avg Likes": a["avg_likes"],
                "Avg Comments": a["avg_comments"],
                "Engagement": f"{a['engagement_label']} ({a['engagement_ratio']:.3f})",
                "Channel URL": a["url"],

                # Tier 1
                "Fit Score": fit,
                "Sponsorship Readiness": calc_sponsorship_readiness(a["engagement_ratio"], a["uploads_90d"], a["inactive_days"]),
                "Cost Efficiency": cost_eff,

                # Tier 2
                "Growth Momentum": calc_growth_momentum(a["views_list"]),
                "Sponsor Saturation": calc_sponsor_saturation(a["video_titles"]),
                "Audience Trust": calc_audience_trust(a["engagement_ratio"], a["views_list"]),

                # Tier 3
                "Language Purity": detected_purity,
                "Product Compatibility": calc_product_compat(a["channel_type"], product_eval),
                "Risk Flags": calc_risk_flags(a["engagement_ratio"], a["inactive_days"], a["uploads_90d"], a["views_list"]),
            }

            df_eval = pd.DataFrame([row])

            # TABLE 1
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 1 — Base Channel Metrics")
            base_cols = ["Channel", "Type", "Subscribers", "Avg Views", "Avg Likes", "Avg Comments", "Engagement", "Channel URL"]
            st.dataframe(df_eval[base_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 2
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 2 — Tier 1")
            t1_cols = ["Channel", "Type", "Fit Score", "Sponsorship Readiness", "Cost Efficiency"]
            st.dataframe(df_eval[t1_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 3
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 3 — Tier 2")
            t2_cols = ["Channel", "Type", "Growth Momentum", "Sponsor Saturation", "Audience Trust"]
            st.dataframe(df_eval[t2_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # TABLE 4
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("Table 4 — Tier 3")
            t3_cols = ["Channel", "Type", "Language Purity", "Product Compatibility", "Risk Flags"]
            st.dataframe(df_eval[t3_cols], use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Optional: recent titles
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

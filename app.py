import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Marketing Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

/* Remove Streamlit UI */
header, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* App background */
.stApp {
    background: url("background.png") no-repeat center center fixed;
    background-size: cover;
}

/* Remove all default paddings */
.block-container {
    padding: 0 !important;
    margin: 0 !important;
}

/* Center main content */
.center-box {
    max-width: 900px;
    margin: 120px auto 0 auto;
    background: rgba(255,255,255,0.92);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
}

/* Buttons */
.big-btn button {
    width: 320px;
    height: 60px;
    font-size: 20px;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = None

# ---------------- HOME ----------------
if st.session_state.mode is None:
    st.markdown("""
    <div class="center-box">
        <h1>📺 YouTube Marketing Investment Intelligence Platform</h1>
        <p>Select one option to continue</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Discover Channels", key="discover"):
            st.session_state.mode = "discover"
            st.rerun()

    with col2:
        if st.button("🔍 Evaluate a Channel", key="evaluate"):
            st.session_state.mode = "evaluate"
            st.rerun()

# ---------------- DISCOVER CHANNELS ----------------
elif st.session_state.mode == "discover":
    st.markdown("""
    <div class="center-box">
        <h2>🚀 Discover Channels</h2>
    </div>
    """, unsafe_allow_html=True)

    country = st.selectbox("Country", ["USA", "India"])
    state = st.text_input("State")
    language = st.selectbox("Language", ["English", "Hindi", "Telugu", "Spanish"])
    product = st.text_input("Marketing Product")
    min_subs = st.number_input("Minimum Subscribers", min_value=0, step=10000)

    if st.button("Find Channels"):
        st.success("🔍 Checking YouTube… Results on the way")

# ---------------- EVALUATE CHANNEL ----------------
elif st.session_state.mode == "evaluate":
    st.markdown("""
    <div class="center-box">
        <h2>🔍 Evaluate a Channel</h2>
    </div>
    """, unsafe_allow_html=True)

    channel = st.text_input("Channel Name or URL")

    if st.button("Evaluate Channel"):
        st.success("📊 Evaluating channel performance…")


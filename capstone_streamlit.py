"""
capstone_streamlit.py — MediCare Hospital Assistant UI
Agentic AI Capstone 2026 | Dr. Kanthi Kiran Sirra

Run with:  streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
import time
import os
import base64
from typing import TypedDict, List, Optional

from agent import build_app, CapstoneState

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediCare Hospital Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #eaf4fb 0%, #f0f7ee 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d6e 0%, #1565c0 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown a { color: #90caf9 !important; }

    /* Header area */
    .medicare-header {
        background: linear-gradient(135deg, #0b3d6e, #1565c0);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
    }
    .medicare-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        margin: 0;
        color: white;
    }
    .medicare-header p {
        margin: 6px 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 14px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    /* User message */
    [data-testid="stChatMessageContent"][aria-label*="user"] {
        background: #e3f2fd !important;
        border-left: 4px solid #1565c0 !important;
    }

    /* Assistant message */
    [data-testid="stChatMessageContent"][aria-label*="assistant"] {
        background: #ffffff !important;
        border-left: 4px solid #2e7d32 !important;
    }

    /* Topic badges */
    .topic-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        color: white !important;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px 2px;
        font-size: 11px;
        font-weight: 500;
    }

    /* Emergency badge */
    .emergency-badge {
        background: #e53935;
        color: white !important;
        border-radius: 8px;
        padding: 6px 14px;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
        font-size: 13px;
    }

    /* Metrics in sidebar */
    .metric-box {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
    }

    /* Main body buttons (Quick Questions) */
    .stButton > button {
        background: white !important;
        color: #1565c0 !important;
        border: 1px solid #1565c0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background: #1565c0 !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }

    /* Sidebar New conversation button */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.28) !important;
        color: white !important;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 14px !important;
        border: 2px solid #1565c0 !important;
        box-shadow: 0 4px 16px rgba(21,101,192,0.12) !important;
    }

    /* Welcome message */
    .welcome-card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        border-left: 5px solid #1565c0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* Faithfulness score */
    .faith-score {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CACHED RESOURCE — load once
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_app():
    """Build app once; cache for all reruns."""
    return build_app()


app, embedder, collection = get_app()


# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 **Welcome to MediCare Hospital!**\n\nI am your 24/7 intelligent assistant. How can I help you today?\n\nYou can ask me about:\n- 📅 OPD Timings & Doctor Schedules\n- 💳 Consultation Fees & Insurance\n- 💊 Pharmacy & Lab Tests\n\n*⚠️ For medical emergencies, please call **040-6600-9999** immediately.*"
    }]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediCare General Hospital")
    st.markdown("*Banjara Hills, Hyderabad — 350 beds*")
    st.markdown("---")

    st.markdown("**📚 Topics I can help with:**")
    topics = [
        "OPD Timings", "Doctor Schedules", "Consultation Fees",
        "Appointment Booking", "Insurance & Cashless",
        "Emergency Services", "Pharmacy", "Lab & Radiology",
        "Health Packages", "Hospital Location & Parking"
    ]
    badges_html = "".join([f'<span class="topic-badge">{t}</span>' for t in topics])
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="emergency-badge">🆘 Emergency: 040-6600-9999</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="metric-box">📞 Helpline: 040-6600-1234</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-box">🌐 www.medicare-hyd.in</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Session Info**")
    st.markdown(f'<div class="metric-box">💬 Turns: {st.session_state.turn_count}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box">🔑 ID: {st.session_state.thread_id[:12]}...</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.thread_id  = str(uuid.uuid4())
        st.session_state.turn_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown("*Powered by LangGraph + ChromaDB*")
    st.markdown("*Agentic AI Course 2026*")


# ─────────────────────────────────────────────────────────────────
# CHAT INPUT RESOLUTION & QUERY PARAMS
# ─────────────────────────────────────────────────────────────────
quick_prompt = None
if "query" in st.query_params:
    quick_prompt = "I need information or a consultation for the " + st.query_params["query"] + " department."
    st.query_params.clear()

user_input = st.chat_input("Ask me anything about MediCare Hospital...")
prompt = user_input or quick_prompt

is_start = len(st.session_state.messages) == 1 and not prompt

# ─────────────────────────────────────────────────────────────────
# MAIN HEADER & DASHBOARD
# ─────────────────────────────────────────────────────────────────
if is_start:
    st.markdown("""
    <style>
    .dash-banner {
        background: linear-gradient(100deg, #fceadb 0%, #fae0c8 100%);
        border-radius: 16px;
        padding: 30px;
        position: relative;
        overflow: hidden;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .dash-banner h1 {
        font-family: 'DM Sans', sans-serif;
        font-size: 32px;
        color: #1c2b33;
        margin: 0;
        font-weight: 800;
        line-height: 1.2;
    }
    .dash-banner p {
        color: #555;
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    .consult-btn {
        background: #1c2b33;
        color: #fff;
        padding: 10px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
    }
    .consult-btn:hover { color: #fceadb; }
    
    .grid-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    .spec-card {
        background: #ffffff;
        border: 1px solid #f1f1f1;
        border-radius: 12px;
        padding: 10px;
        display: flex;
        align-items: center;
        text-decoration: none !important;
        color: #333 !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .spec-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
        border-color: #d1dde6;
    }
    .icon-box {
        background: #f6f8fb;
        border-radius: 8px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 20px;
    }
    .spec-text {
        font-weight: 600;
        font-size: 13px;
        line-height: 1.2;
    }
    </style>
    
<div class="dash-banner">
<div>
<h1>Talk to a Doctor for an Instant advice</h1>
<p>Get 5% Off | Use Code <b>CC50</b></p>
<a href="?query=General+Physician" target="_self" class="consult-btn">Consult Now</a>
</div>
<div style="font-size: 60px;">🩺💊</div>
</div>
    
<h3 style="margin-top:20px; font-weight:700;">Browse by Specialties</h3>
    
<div class="grid-container">
<a href="?query=General+Physician" target="_self" class="spec-card">
    <div class="icon-box">🩺</div><div class="spec-text">General<br>Physician</div>
</a>
<a href="?query=Dermatology" target="_self" class="spec-card">
    <div class="icon-box">🧴</div><div class="spec-text">Dermatology</div>
</a>
<a href="?query=Obstetrics+and+Gynecology" target="_self" class="spec-card">
    <div class="icon-box">🤰</div><div class="spec-text">Obstetrics &<br>Gynecology</div>
</a>
<a href="?query=Orthopaedics" target="_self" class="spec-card">
    <div class="icon-box">🦴</div><div class="spec-text">Orthopaedics</div>
</a>
<a href="?query=ENT" target="_self" class="spec-card">
    <div class="icon-box">👂</div><div class="spec-text">ENT</div>
</a>
<a href="?query=Neurology" target="_self" class="spec-card">
    <div class="icon-box">🧠</div><div class="spec-text">Neurology</div>
</a>
<a href="?query=Cardiology" target="_self" class="spec-card">
    <div class="icon-box">❤️</div><div class="spec-text">Cardiology</div>
</a>
<a href="?query=Urology" target="_self" class="spec-card">
    <div class="icon-box">💧</div><div class="spec-text">Urology</div>
</a>
<a href="?query=Gastroenterology" target="_self" class="spec-card">
    <div class="icon-box">🍽️</div><div class="spec-text">Gastroenterology</div>
</a>
<a href="?query=Psychiatry" target="_self" class="spec-card">
    <div class="icon-box">🗣️</div><div class="spec-text">Psychiatry</div>
</a>
<a href="?query=Paediatrics" target="_self" class="spec-card">
    <div class="icon-box">👶</div><div class="spec-text">Paediatrics</div>
</a>
<a href="?query=Pulmonology" target="_self" class="spec-card">
    <div class="icon-box">🫁</div><div class="spec-text">Pulmonology</div>
</a>
<a href="?query=Endocrinology" target="_self" class="spec-card">
    <div class="icon-box">🦠</div><div class="spec-text">Endocrinology</div>
</a>
<a href="?query=Nephrology" target="_self" class="spec-card">
    <div class="icon-box">⚕️</div><div class="spec-text">Nephrology</div>
</a>
<a href="?query=Neurosurgery" target="_self" class="spec-card">
    <div class="icon-box">🔪</div><div class="spec-text">Neurosurgery</div>
</a>
<a href="?query=Rheumatology" target="_self" class="spec-card">
    <div class="icon-box">🦵</div><div class="spec-text">Rheumatology</div>
</a>
<a href="?query=Ophthalmology" target="_self" class="spec-card">
    <div class="icon-box">👁️</div><div class="spec-text">Ophthalmology</div>
</a>
<a href="?query=Surgical+Gastro" target="_self" class="spec-card">
    <div class="icon-box">🩺</div><div class="spec-text">Surgical<br>Gastro</div>
</a>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="medicare-header">
        <h1>🏥 MediCare Hospital Assistant</h1>
        <p>Your intelligent 24/7 patient assistant — OPD timings, appointments, fees, insurance & more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────
    # CHAT HISTORY DISPLAY
    # ─────────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if "Welcome to MediCare Hospital" in msg["content"]:
            continue
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
            st.write(msg["content"])

if is_start and not prompt:
    # Render animated nurse right-bottom popup only if no prompt is being processed
    if os.path.exists("nurse.png"):
        with open("nurse.png", "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .nurse-popup {{
            position: fixed;
            bottom: 110px;
            right: 30px;
            display: flex;
            align-items: flex-end;
            gap: 15px;
            z-index: 99999;
            animation: slideInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        @keyframes slideInUp {{
            0% {{ transform: translateY(150px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}
        .nurse-popup .speech-bubble {{
            background: #ffffff;
            color: #0b3d6e;
            padding: 12px 18px;
            border-radius: 20px;
            border-bottom-right-radius: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            font-weight: 600;
            font-size: 14px;
            max-width: 220px;
            animation: floatP 3s ease-in-out infinite;
        }}
        @keyframes floatP {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-5px); }}
            100% {{ transform: translateY(0px); }}
        }}
        .nurse-popup img {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 4px solid #ffffff;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            object-fit: cover;
        }}
        </style>
        <div class="nurse-popup">
            <div class="speech-bubble">Got questions? Try asking me about OPD timings or Doctors below! 👇</div>
            <img src="data:image/png;base64,{b64_img}" />
        </div>
        """, unsafe_allow_html=True)

if prompt:
    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Typing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = CapstoneState(
                question=prompt,
                messages=[],
                route="",
                retrieved="",
                sources=[],
                tool_result="",
                answer="",
                faithfulness=0.0,
                eval_retries=0,
                user_name=None
            )
            result = app.invoke(initial_state, config=config)

        answer      = result["answer"]
        sources     = result.get("sources", [])
        faithfulness = result.get("faithfulness", 1.0)
        route       = result.get("route", "")

        def stream_data(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.015)
                
        st.write_stream(stream_data(answer))

    # Update session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "faithfulness": faithfulness
    })
    st.session_state.turn_count += 1

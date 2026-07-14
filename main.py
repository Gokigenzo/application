import streamlit as st
from loguru import logger
from app.utils.logging import setup_logging
from app.streamlit_ui.router import Router
from app.streamlit_ui.components import inject_premium_css
from app.streamlit_ui.pages import (
    dashboard, camera_realtime, camera_capture, 
    students, sessions, attendance, monitoring, settings
)
from app.streamlit_ui.dependencies import get_services

# Initialize Loguru configuration
setup_logging()

# Configure page setup
st.set_page_config(
    page_title="AI Face Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ingest aesthetic premium CSS styles
inject_premium_css()

# Sidebar Brand Header
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-bottom: 25px; margin-top: 10px;">
        <h1 style="color: #6366f1; font-size: 1.75rem; font-weight: 700; margin-bottom: 2px; letter-spacing: -0.04em;">🎓 AutoAttendance</h1>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">AI Face Recognition System</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Navigation")
current_page = Router.get_current_page()

# Registry mapping page keys to modules
pages = {
    "Dashboard": dashboard,
    "Real-time Camera": camera_realtime,
    "Capture Attendance": camera_capture,
    "Student Management": students,
    "Attendance Logs": attendance,
    "Session Management": sessions,
    "System Monitor": monitoring,
    "Settings": settings
}

# Sidebar radio selector
selected = st.sidebar.radio(
    "Navigation Link",
    options=list(pages.keys()),
    index=list(pages.keys()).index(current_page),
    label_visibility="collapsed"
)

# Execute route transition if selection changes
if selected != current_page:
    Router.set_page(selected)

st.sidebar.markdown("---")

# Render active session indicator badge in sidebar
try:
    services = get_services()
    active_sessions = services["session_service"].get_active_sessions()
    
    if active_sessions:
        s = active_sessions[0]
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; font-size: 0.75rem; color: #10b981; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">● Active Session</div>
                <div style="font-size: 0.8rem; color: #e2e8f0; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{s.name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; font-size: 0.75rem; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em;">● No Active Session</div>
            </div>
            """,
            unsafe_allow_html=True
        )
except Exception as ex:
    logger.error(f"Error drawing sidebar status widget: {str(ex)}")

st.sidebar.markdown(
    """
    <div style="position: fixed; bottom: 15px; width: 210px; font-size: 0.75rem; color: #475569; text-align: center;">
        v1.0.0-prod | OS: Linux<br>
        © 2026 AI Engineering Team
    </div>
    """,
    unsafe_allow_html=True
)

# Render selected page view
try:
    pages[current_page].show()
except Exception as e:
    logger.exception("Application crash caught in main Streamlit renderer loop")
    st.error(f"🚨 An error occurred while loading this page: {str(e)}")

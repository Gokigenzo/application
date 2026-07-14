import streamlit as st
from datetime import datetime, timedelta, timezone
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services
from app.core.exceptions import SessionException

def show() -> None:
    """Renders the Session Management page UI."""
    services = get_services()
    session_service = services["session_service"]

    draw_header_with_badge("Session Management", is_active=False, badge_text="Start & Close Lectures")

    active_sessions = session_service.get_active_sessions()
    all_sessions = session_service.list_sessions()

    # --- Section 1: Active Sessions ---
    st.subheader("Active Attendance Sessions")
    if active_sessions:
        for s in active_sessions:
            # Display active session details inside a styled container
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 15px;">
                    <div style="font-weight: 700; font-size: 1.1rem; color: #10b981; margin-bottom: 5px;">📖 {s.name}</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1;">
                        Teacher: <strong>{s.teacher_name}</strong> | 
                        Started At: <strong>{s.start_time.astimezone().strftime('%H:%M:%S')}</strong> | 
                        Ends At: <strong>{s.end_time.astimezone().strftime('%H:%M:%S')}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Trigger to close session
            if st.button(f"🛑 Close Session: {s.name}", key=f"close_{s.id}", type="primary"):
                session_service.close_session(s.id)
                st.success("Session closed successfully.")
                st.rerun()
    else:
        st.info("No active lecture sessions are running currently. Create one below to begin recording attendance.")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- Section 2: Start Session Form ---
    st.subheader("Start a New Session")
    
    with st.form("start_session_form"):
        name = st.text_input("Session Title / Class Name", placeholder="Physics 102 - Light & Waves")
        teacher_name = st.text_input("Instructor / Teacher Name", placeholder="Prof. Richard Feynman")
        
        col1, col2 = st.columns(2)
        with col1:
            duration_min = st.number_input("Session Duration (minutes)", min_value=5, max_value=360, value=60, step=5)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="color: #64748b; font-size: 0.8rem; line-height: 1.4;">'
                'Once started, students will be able to check in via the Real-time Camera or Capture Mode.</div>',
                unsafe_allow_html=True
            )

        submit = st.form_submit_button("🚀 Start Session", type="primary")

        if submit:
            if not name or not teacher_name:
                st.error("Missing Input: Please enter both a Session Title and Instructor Name.")
            elif active_sessions:
                st.error("Conflict: An active session is already running! Please close it before starting a new one.")
            else:
                try:
                    start_time = datetime.now(timezone.utc)
                    end_time = start_time + timedelta(minutes=int(duration_min))
                    
                    # Create active session
                    session_service.create_session(
                        name=name,
                        teacher_name=teacher_name,
                        start_time=start_time,
                        end_time=end_time
                    )
                    st.success(f"Session '{name}' is now active and ready for check-ins.")
                    st.rerun()
                except SessionException as e:
                    st.error(f"Failed to start session: {e.message}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- Section 3: History Archive ---
    st.subheader("Archived Sessions")
    archived = [s for s in all_sessions if not s.is_active]

    if archived:
        archived_data = [{
            "Session Title": s.name,
            "Instructor": s.teacher_name,
            "Date": s.start_time.astimezone().strftime("%Y-%m-%d"),
            "Started At": s.start_time.astimezone().strftime("%H:%M"),
            "Ended At": s.end_time.astimezone().strftime("%H:%M")
        } for s in archived]
        
        df_archived = pd.DataFrame(archived_data)
        st.dataframe(df_archived, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div style="color: #64748b; font-style: italic;">No past archived sessions found.</div>',
            unsafe_allow_html=True
        )

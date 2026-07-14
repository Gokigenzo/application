import streamlit as st
import pandas as pd
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services

def show() -> None:
    """Renders the Attendance Management page UI."""
    services = get_services()
    session_service = services["session_service"]
    attendance_service = services["attendance_service"]
    student_service = services["student_service"]

    draw_header_with_badge("Attendance Management", is_active=False, badge_text="Logs & Export")

    sessions = session_service.list_sessions()
    students = student_service.list_students()
    total_students_enrolled = len(students)

    if not sessions:
        st.info("No lecture sessions found. Create a session under 'Sessions' and record attendance first.")
        return

    # Dropdown to filter attendance by session
    st.subheader("Select Session")
    selected_session = st.selectbox(
        "Choose Class Session",
        options=sessions,
        format_func=lambda s: f"{s.name} (Date: {s.start_time.astimezone().strftime('%Y-%m-%d')} | Instructor: {s.teacher_name})"
    )

    if selected_session:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. Fetch attendance records for this session
        sheet = attendance_service.get_attendance_sheet(selected_session.id)
        present_count = len(sheet)
        
        # Calculate attendance rate
        rate_pct = (present_count / total_students_enrolled * 100.0) if total_students_enrolled > 0 else 0.0

        # Render Stats cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Present Students", value=present_count)
        with col2:
            st.metric(label="Class Roster Enrollment", value=total_students_enrolled)
        with col3:
            st.metric(label="Attendance Rate", value=f"{rate_pct:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Attendance Roster")

        if sheet:
            df = pd.DataFrame(sheet)
            df["marked_at"] = pd.to_datetime(df["marked_at"]).dt.tz_convert("Asia/Bangkok").dt.strftime("%Y-%m-%d %H:%M:%S")

            # Format dataframe for display
            df_display = df[["student_id", "student_name", "student_email", "marked_at", "status"]].rename(
                columns={
                    "student_id": "Student Code",
                    "student_name": "Student Name",
                    "student_email": "Email",
                    "marked_at": "Checked In At",
                    "status": "Status"
                }
            )

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Export to CSV section
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = df_display.to_csv(index=False).encode("utf-8")
            
            filename = f"attendance_{selected_session.name.lower().replace(' ', '_')}_{selected_session.start_time.strftime('%Y%m%d')}.csv"
            
            st.download_button(
                label="📥 Export Attendance Sheet to CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No attendance records recorded for this session yet.")

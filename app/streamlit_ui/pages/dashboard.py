import streamlit as st
import pandas as pd
import plotly.express as px
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services

def show() -> None:
    """Renders the Dashboard page UI."""
    services = get_services()
    student_service = services["student_service"]
    session_service = services["session_service"]
    attendance_service = services["attendance_service"]

    draw_header_with_badge("Dashboard", is_active=True, badge_text="System Online")

    # 1. Fetch system statistics
    students = student_service.list_students()
    active_sessions = session_service.get_active_sessions()
    all_sessions = session_service.list_sessions()
    all_attendance = attendance_service.attendance_repo.list_all()

    # 2. Render Metric cards in a grid layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Registered Students", value=len(students))
    with col2:
        st.metric(label="Active Sessions", value=len(active_sessions))
    with col3:
        st.metric(label="Total Attendance Logs", value=len(all_attendance))

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Analytics visualization: Plotly Attendance Chart
    st.subheader("Attendance by Lecture Session")
    if all_sessions:
        data = []
        for s in all_sessions:
            recs = attendance_service.get_attendance_sheet(s.id)
            data.append({
                "Session Title": s.name,
                "Present Students": len(recs),
                "Status": "Active" if s.is_active else "Archived"
            })
        df = pd.DataFrame(data)
        
        # Color coding: Green for active sessions, indigo/purple for closed sessions
        fig = px.bar(
            df,
            x="Session Title",
            y="Present Students",
            color="Status",
            color_discrete_map={"Active": "#10b981", "Archived": "#6366f1"},
            template="plotly_dark",
            barmode="group"
        )
        
        # Style chart with transparent backgrounds to fit glassmorphism theme
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Outfit",
            font_color="#94a3b8",
            xaxis_title="Lecture Session",
            yaxis_title="Total Present Students",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No lecture sessions found. Go to 'Sessions' on the navigation menu to create a class session.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Recent Attendance log feed
    st.subheader("Recent Attendance Records")
    recent_records = []
    
    # Query logs from active sessions first, fallback to all logs
    for s in active_sessions:
        recent_records.extend(attendance_service.get_attendance_sheet(s.id))
        
    if not recent_records and all_sessions:
        for s in all_sessions[:3]:  # Grab logs from last 3 sessions
            recent_records.extend(attendance_service.get_attendance_sheet(s.id))

    if recent_records:
        # Take the top 5 most recent records
        recent_records = recent_records[:5]
        df_recent = pd.DataFrame(recent_records)
        
        # Format dates for clean rendering
        df_recent["marked_at"] = pd.to_datetime(df_recent["marked_at"]).dt.tz_convert("Asia/Bangkok").dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(
            df_recent[["student_id", "student_name", "marked_at", "status"]].rename(
                columns={
                    "student_id": "Student Code",
                    "student_name": "Student Name",
                    "marked_at": "Check-in Time",
                    "status": "Attendance status"
                }
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.markdown(
            '<div style="color: #64748b; font-style: italic;">No attendance check-ins recorded yet. Start a session and open the Real-time Camera page.</div>',
            unsafe_allow_html=True
        )

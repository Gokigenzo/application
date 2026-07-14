import streamlit as st
import cv2
import numpy as np
import uuid
import pandas as pd
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services
from app.core.exceptions import AttendanceException

def show() -> None:
    """Renders the Capture Attendance page UI."""
    services = get_services()
    session_service = services["session_service"]
    attendance_service = services["attendance_service"]
    ai_service = services["ai_service"]

    draw_header_with_badge("Capture Attendance", is_active=False, badge_text="Mode 2 - Fallback")

    # Safeguard check for active lecture session
    active_sessions = session_service.get_active_sessions()
    if not active_sessions:
        st.warning("⚠️ No active lecture session found! Please start a session first under 'Sessions' page.")
        return

    active_sess = active_sessions[0]
    st.info(f"📸 Taking snapshot attendance for session: **{active_sess.name}**")

    # Streamlit camera widget
    img_file = st.camera_input("Snapshot the classroom")

    if img_file is not None:
        # Convert captured file bytes to cv2 BGR image
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if cv_img is None:
            st.error("Failed to decode image from camera. Please try again.")
            return

        with st.spinner("Analyzing snapshot faces..."):
            # Run immediate recognition without tracker/voters
            results = ai_service.recognize_snapshot(cv_img)

        if not results:
            st.warning("No faces detected in the snapshot. Make sure the lighting is adequate and students are facing the camera.")
            return

        # Prepare image copy for overlays drawing
        annotated_img = cv_img.copy()
        
        recognized_students = []
        unknown_count = 0
        marked_names = []

        # Iterate and mark attendance
        for res in results:
            x1, y1, x2, y2 = map(int, res["bbox"])
            
            if res["is_recognized"]:
                student_uuid_str = res["student_uuid"]
                student_uuid = uuid.UUID(student_uuid_str)
                name = res["name"]
                code = res["student_code"]
                score = res["score"]

                # Record in database
                try:
                    status = attendance_service.record_attendance(active_sess.id, student_uuid)
                    if status == "RECORDED":
                        marked_names.append(f"**{name}** (New)")
                    elif status == "ALREADY_ATTENDED":
                        marked_names.append(f"{name} (Already marked)")
                except AttendanceException as e:
                    st.error(f"Error marking attendance for {name}: {e.message}")

                recognized_students.append({
                    "Name": name,
                    "Code": code,
                    "Match Score": f"{score:.2%}"
                })
                
                # Green box for recognized student
                color = (16, 185, 129)
                label = f"{name} ({score:.1%})"
            else:
                unknown_count += 1
                # Blue box for unknown face
                color = (59, 130, 246)
                label = "Unknown"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Render the annotated image
        st.subheader("Analysis Output")
        st.image(annotated_img, channels="BGR", use_container_width=True)

        # Display results summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Faces Recognized", value=len(recognized_students))
        with col2:
            st.metric(label="Unknown Faces", value=unknown_count)

        if marked_names:
            st.success(f"Processed attendance: {', '.join(marked_names)}")

        if recognized_students:
            st.markdown("### Match Details")
            st.table(pd.DataFrame(recognized_students))

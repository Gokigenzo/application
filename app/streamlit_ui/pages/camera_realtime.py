import queue
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services
from app.core.exceptions import AttendanceException

class VideoProcessor(VideoProcessorBase):
    """Processes video frames, drawing track overlays and queuing attendance events."""
    
    def __init__(self, ai_service, active_session_id, attendance_queue):
        self.ai_service = ai_service
        self.active_session_id = active_session_id
        self.attendance_queue = attendance_queue
        # Prevent queueing the same track ID multiple times in this stream run
        self.marked_track_ids = set()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Decode AV Frame to BGR NumPy array
        img = frame.to_ndarray(format="bgr24")

        # 2. Run face detection, tracking, and voting matching
        try:
            tracks = self.ai_service.process_frame(img)
        except Exception as e:
            # Return unchanged frame on AI exception to avoid freezing the video feed
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 3. Draw tracking overlays
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Green border for identified students, red border for unknown faces
            color = (16, 185, 129) if track.is_recognized else (59, 130, 246)  # Green vs Blue
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label name above box
            label = f"{track.identified_name} (ID: {track.track_id})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 4. Queue attendance events for recognized students
            if track.is_recognized and track.track_id not in self.marked_track_ids:
                # Put tuple: (session_uuid, student_uuid, student_name)
                self.attendance_queue.put((self.active_session_id, track.identified_uuid, track.identified_name))
                self.marked_track_ids.add(track.track_id)

        # 5. Return updated BGR frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def show() -> None:
    """Renders the Real-time Camera page UI."""
    services = get_services()
    session_service = services["session_service"]
    attendance_service = services["attendance_service"]
    ai_service = services["ai_service"]

    draw_header_with_badge("Real-time Attendance", is_active=True, badge_text="Mode 1")

    # Ensure an active session is running before booting up the camera
    active_sessions = session_service.get_active_sessions()
    if not active_sessions:
        st.warning("⚠️ No active lecture session found! Please start a session first under 'Sessions' page before opening the camera.")
        return

    active_sess = active_sessions[0]
    st.info(f"📹 Recording attendance for: **{active_sess.name}** (Instructor: *{active_sess.teacher_name}*)")

    # Initialize thread-safe attendance queue if not present in st.session_state
    if "attendance_queue" not in st.session_state:
        st.session_state.attendance_queue = queue.Queue()

    # Process events queued by the WebRTC background threads
    q = st.session_state.attendance_queue
    processed_count = 0
    while not q.empty():
        sess_id, student_uuid, name = q.get()
        try:
            status = attendance_service.record_attendance(sess_id, student_uuid)
            if status == "RECORDED":
                st.toast(f"🎉 Marked Present: **{name}**!", icon="✅")
                processed_count += 1
            elif status == "ALREADY_ATTENDED":
                pass  # Ignore duplicates silently
        except AttendanceException as e:
            st.error(f"Failed to record attendance: {e.message}")

    # WebRTC Streamer element
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    webrtc_streamer(
        key="attendance-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(
            ai_service=ai_service,
            active_session_id=active_sess.id,
            attendance_queue=st.session_state.attendance_queue
        ),
        rtc_configuration={
            # Public STUN servers to bypass NAT routers in remote deployment environments
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Render a refreshing roster of students checked-in
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Attendance Log (Current Session)")
    with col2:
        if st.button("🔄 Refresh Check-ins", use_container_width=True):
            st.rerun()

    sheet = attendance_service.get_attendance_sheet(active_sess.id)
    if sheet:
        # Convert sheet to pandas dataframe for clean display
        df = pd.DataFrame(sheet)
        df["marked_at"] = pd.to_datetime(df["marked_at"]).dt.tz_convert("Asia/Bangkok").dt.strftime("%H:%M:%S")
        st.dataframe(
            df[["student_id", "student_name", "marked_at", "status"]].rename(
                columns={
                    "student_id": "Student Code",
                    "student_name": "Student Name",
                    "marked_at": "Marked At",
                    "status": "Status"
                }
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No students checked in yet. Faces will be automatically processed once the camera starts.")

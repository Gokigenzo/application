import streamlit as st
import cv2
import numpy as np
import pandas as pd
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services
from app.core.exceptions import DatabaseException, AIException

def show() -> None:
    """Renders the Student Management page UI."""
    services = get_services()
    student_service = services["student_service"]

    draw_header_with_badge("Student Management", is_active=False, badge_text="Roster & Registration")

    # Layout: Two tabs - Register Student & Student Directory
    tab_list, tab_reg = st.tabs(["📋 Student Directory", "👤 Register Student"])

    # --- TAB 1: Student Directory ---
    with tab_list:
        st.subheader("Registered Students")
        students = student_service.list_students()

        if students:
            # Formulate list to DataFrame
            student_data = [{
                "UUID": str(s.id),
                "Student ID": s.student_id,
                "Name": s.name,
                "Email": s.email if s.email else "N/A",
                "Created At": s.created_at.strftime("%Y-%m-%d")
            } for s in students]
            df = pd.DataFrame(student_data)

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Delete student panel
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Delete Student")
            with st.form("delete_student_form"):
                student_to_delete = st.selectbox(
                    "Select Student to Delete",
                    options=students,
                    format_func=lambda s: f"{s.name} ({s.student_id})"
                )
                submit_delete = st.form_submit_button("🗑️ Delete Student", type="primary")

                if submit_delete and student_to_delete:
                    # Execute deletion
                    success = student_service.delete_student(student_to_delete.id)
                    if success:
                        st.success(f"Successfully deleted student: **{student_to_delete.name}**.")
                        st.rerun()
                    else:
                        st.error("Failed to delete student.")
        else:
            st.info("No students registered yet. Switch to the 'Register Student' tab to enroll your first student.")

    # --- TAB 2: Register Student ---
    with tab_reg:
        st.subheader("New Student Enrollment")
        
        with st.form("register_student_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", placeholder="Jane Doe")
                student_code = st.text_input("Student Code / Registration ID", placeholder="STD202604")
            with col2:
                email = st.text_input("Email (Optional)", placeholder="jane.doe@school.edu")

            st.markdown("#### Face Enrollment Photos")
            st.markdown(
                '<div style="color: #64748b; font-size: 0.85rem; margin-bottom: 15px;">'
                'Provide face photos of the student. For optimal accuracy, upload or snap 1 to 3 photos '
                'with clear lighting, centered position, and no face coverings.</div>',
                unsafe_allow_html=True
            )

            # Enrollment photo input selectors
            upload_files = st.file_uploader(
                "Upload Enrollment Images", 
                type=["png", "jpg", "jpeg"], 
                accept_multiple_files=True
            )
            
            camera_img = st.camera_input("Capture Enrollment Photo (Alternative)")

            submit_reg = st.form_submit_button("➕ Register Student & Process Embeddings", type="primary")

            if submit_reg:
                if not name or not student_code:
                    st.error("Required Fields Missing: Please provide both Name and Student Code.")
                else:
                    images = []

                    # 1. Process uploaded files
                    if upload_files:
                        for uploaded_file in upload_files:
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            if img_bgr is not None:
                                images.append(img_bgr)

                    # 2. Process camera photo
                    if camera_img:
                        file_bytes = np.frombuffer(camera_img.getvalue(), np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if img_bgr is not None:
                            images.append(img_bgr)

                    if not images:
                        st.error("Missing Face Photos: Please take a photo or upload at least one image file.")
                    else:
                        try:
                            with st.spinner("Processing facial landmarks and extracting embeddings..."):
                                student_service.register_student(
                                    student_id=student_code,
                                    name=name,
                                    email=email if email else None,
                                    face_images=images
                                )
                            st.success(f"🎉 Successfully enrolled student: **{name}** (Code: {student_code})!")
                            st.balloons()
                            st.rerun()
                        except DatabaseException as e:
                            st.error(f"Database Error: {e.message}")
                        except AIException as e:
                            st.error(f"AI Processing Error: {e.message}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")

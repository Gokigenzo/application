import streamlit as st
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services
from app.core.configs import get_settings

def show() -> None:
    """Renders the Settings page UI."""
    services = get_services()
    ai_service = services["ai_service"]
    cache = services["cache"]
    
    draw_header_with_badge("Settings", is_active=False, badge_text="Parameters & Configurations")

    st.subheader("Face Recognition Parameters")
    
    with st.form("settings_form"):
        # 1. Similarity Threshold
        similarity_threshold = st.slider(
            "Cosine Similarity Threshold (Recognition)",
            min_value=0.10,
            max_value=0.95,
            value=float(ai_service.similarity_threshold),
            step=0.05,
            help="Minimum cosine similarity matching score for a face. Higher values decrease false positives but increase false negatives."
        )

        # 2. Detection Threshold
        detection_threshold = st.slider(
            "SCRFD Confidence Threshold (Detection)",
            min_value=0.10,
            max_value=0.95,
            value=float(ai_service.detector.threshold),
            step=0.05,
            help="Confidence threshold for face bounding box predictions."
        )

        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("Temporal Voting Settings")
        
        # 3. Buffer sizes and votes counts
        col1, col2 = st.columns(2)
        with col1:
            buffer_size = st.number_input(
                "Voter Buffer Size (Frames)",
                min_value=1,
                max_value=50,
                value=int(ai_service.buffer_size),
                step=1,
                help="Number of consecutive tracking frames used to evaluate the temporal identity voting pool."
            )
        with col2:
            min_votes = st.number_input(
                "Required Votes for Check-in",
                min_value=1,
                max_value=50,
                value=int(ai_service.min_votes),
                step=1,
                help="Minimum number of matching votes in the buffer required to lock the student's identity."
            )

        submit = st.form_submit_button("💾 Save Settings", type="primary")

        if submit:
            # Validate voting parameters consistency
            if min_votes > buffer_size:
                st.error("Invalid Configuration: Required votes cannot exceed the voter buffer size.")
            else:
                # Update AI Service settings dynamically
                ai_service.similarity_threshold = similarity_threshold
                ai_service.detector.threshold = detection_threshold
                ai_service.buffer_size = buffer_size
                ai_service.min_votes = min_votes

                st.success("Settings saved successfully! Changes are active in the running computer vision pipeline.")

    # Cache refresh utility
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Database Cache Management")
    st.markdown(
        '<div style="color: #64748b; font-size: 0.85rem; margin-bottom: 15px;">'
        'Force reload the in-memory vector database cache from Supabase. Use this if student '
        'face vectors were modified directly inside PostgreSQL.</div>',
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Reload Embedding Cache", use_container_width=True):
        with st.spinner("Refetching vectors from database..."):
            cache.load_from_db(services["db"])
        st.success(f"Cache reloaded! Loaded **{cache.size}** student face embedding profiles.")

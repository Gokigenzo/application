import streamlit as st

class Router:
    """Manages view routing and session state navigation."""

    @staticmethod
    def get_current_page() -> str:
        """Retrieves the active page name from session state."""
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
        return st.session_state.current_page

    @staticmethod
    def set_page(page_name: str) -> None:
        """Changes the active page and triggers a page rerun.

        Args:
            page_name: The target view key.
        """
        st.session_state.current_page = page_name
        st.rerun()

import streamlit as st

def inject_premium_css() -> None:
    """Injects modern, premium CSS overrides into the Streamlit application.
    
    Includes Google Fonts, glassmorphism cards, glowing active/inactive
    indicators, customized scrollbars, and styled metric display boxes.
    """
    premium_css = """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Global font styles */
        html, body, [class*="css"], .stMarkdown, p, div, span, label, button, select, input {
            font-family: 'Outfit', sans-serif !important;
        }

        /* App container background adjustments */
        .main {
            background-color: #0b0f19;
            color: #f8fafc;
        }
        
        /* Glassmorphism panel cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.6) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4) !important;
            margin-bottom: 20px !important;
            transition: all 0.3s ease-in-out;
        }
        
        .glass-card:hover {
            border: 1px solid rgba(99, 102, 241, 0.25) !important;
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.1) !important;
        }
        
        .glass-card-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
            letter-spacing: -0.02em;
        }
        
        /* Glowing neon dot classes */
        .glow-dot-active {
            width: 11px;
            height: 11px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981, 0 0 20px #059669;
            display: inline-block;
            margin-right: 10px;
        }
        
        .glow-dot-inactive {
            width: 11px;
            height: 11px;
            background-color: #ef4444;
            border-radius: 50%;
            box-shadow: 0 0 10px #ef4444, 0 0 20px #dc2626;
            display: inline-block;
            margin-right: 10px;
        }

        /* Slim, responsive custom scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.3);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.4);
            border-radius: 8px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.75);
        }

        /* Streamlit components overrides */
        div[data-testid="stMetricValue"] {
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -0.03em;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600 !important;
        }
        
        /* Metric block wrapper container */
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px 20px;
        }
    </style>
    """
    st.markdown(premium_css, unsafe_allow_html=True)


def draw_header_with_badge(title: str, is_active: bool = False, badge_text: str = "") -> None:
    """Draws a premium header page layout containing a glowing status indicator."""
    status_class = "glow-dot-active" if is_active else "glow-dot-inactive"
    badge_html = f'<span style="font-size: 0.8rem; background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 4px 10px; border-radius: 20px; margin-left: 15px; border: 1px solid rgba(99, 102, 241, 0.3);">{badge_text}</span>' if badge_text else ""
    
    header_html = f"""
    <div style="display: flex; align-items: center; margin-top: 10px; margin-bottom: 25px;">
        <span class="{status_class}"></span>
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.03em; display: inline-block;">
            {title}
        </h1>
        {badge_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def show_glass_card(title: str, inner_html: str) -> None:
    """Outputs a custom HTML block styled as a glassmorphic container."""
    card_html = f"""
    <div class="glass-card">
        <div class="glass-card-header">{title}</div>
        <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
            {inner_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

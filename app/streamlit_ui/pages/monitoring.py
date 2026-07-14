import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.streamlit_ui.components import draw_header_with_badge
from app.streamlit_ui.dependencies import get_services

def show() -> None:
    """Renders the System Monitor & logs page UI."""
    services = get_services()
    monitor_service = services["monitor_service"]
    cache = services["cache"]

    draw_header_with_badge("System Monitor", is_active=True, badge_text="Live Telemetry")

    # Fetch stats
    sys_metrics = monitor_service.get_system_metrics()
    ai_metrics = monitor_service.get_ai_metrics()

    # Layout: Metrics Tabs (System, AI Models, Logs)
    tab_sys, tab_ai, tab_logs = st.tabs([
        "🖥️ System Resources", 
        "🧠 AI Models & Latency", 
        "📋 Developer Logs"
    ])

    # --- TAB 1: System Resources (CPU, Memory, Cache) ---
    with tab_sys:
        st.subheader("Resource Utilization")
        
        # 1. Gauge indicators using Plotly Graph Objects
        col1, col2 = st.columns(2)
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_metrics["cpu_usage_pct"],
                title={'text': "CPU Load (%)", 'font': {'size': 18, 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#6366f1"},
                    'bgcolor': "rgba(30, 41, 59, 0.3)",
                    'borderwidth': 1,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [50, 85], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_cpu.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Outfit",
                font_color="#f8fafc",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cpu, use_container_width=True)

        with col2:
            fig_ram = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_metrics["memory_usage_pct"],
                title={'text': "RAM Utilization (%)", 'font': {'size': 18, 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#a78bfa"},
                    'bgcolor': "rgba(30, 41, 59, 0.3)",
                    'borderwidth': 1,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [50, 85], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_ram.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Outfit",
                font_color="#f8fafc",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ram, use_container_width=True)

        # Numerical readouts
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                label="Memory Allocated", 
                value=f"{sys_metrics['memory_used_mb']:.1f} MB",
                delta=f"Total: {sys_metrics['memory_total_mb']:.0f} MB",
                delta_color="off"
            )
        with col_m2:
            st.metric(
                label="Local Vector Cache", 
                value=f"{cache.size} profiles",
                delta="NumPy Hot Matrix",
                delta_color="normal"
            )
        with col_m3:
            st.metric(
                label="Disk Capacity Space", 
                value=f"{sys_metrics['disk_usage_pct']:.0f}%",
                delta="Root Partition Usage",
                delta_color="inverse"
            )

    # --- TAB 2: AI Models & Latency (SCRFD, ArcFace, Caching) ---
    with tab_ai:
        st.subheader("Model Inference Runtimes")
        
        # Draw latency comparison chart
        latency_data = {
            "Pipeline Stage": ["Face Detection (SCRFD)", "Feature Extraction (ArcFace)", "Vector Cache Match"],
            "Average Latency (ms)": [
                ai_metrics["avg_detection_ms"],
                ai_metrics["avg_embedding_ms"],
                ai_metrics["avg_lookup_ms"]
            ]
        }
        df_latency = pd.DataFrame(latency_data)

        # Plotly bar chart
        fig_lat = px.bar(
            df_latency,
            x="Pipeline Stage",
            y="Average Latency (ms)",
            text_auto='.1f',
            template="plotly_dark",
            color="Pipeline Stage",
            color_discrete_sequence=["#f43f5e", "#ec4899", "#8b5cf6"]
        )
        
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Outfit",
            font_color="#94a3b8",
            xaxis_title="",
            yaxis_title="Time in Milliseconds (ms)",
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # Camera frame rate readout
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Video Feed Performance")
        st.metric(
            label="Inference Frame Rate", 
            value=f"{ai_metrics['avg_fps']:.1f} FPS", 
            delta="Target: 20-30 FPS"
        )

    # --- TAB 3: Developer Logs (Stdout log viewer) ---
    with tab_logs:
        st.subheader("Application Terminal Logs")
        
        col_l1, col_l2 = st.columns([4, 1])
        with col_l1:
            st.markdown(
                '<div style="color: #64748b; font-size: 0.85rem; margin-top: 5px;">'
                'Showing recent developer debug logs from logs/app.log file.</div>',
                unsafe_allow_html=True
            )
        with col_l2:
            if st.button("🔄 Refresh Logs", use_container_width=True):
                st.rerun()

        logs = monitor_service.get_recent_logs(max_lines=120)
        
        # Render in a clean scrollable box
        st.code("\n".join(logs), language="log")

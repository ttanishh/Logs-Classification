
import streamlit as st
import pandas as pd

from classify_logic import classify
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Log Monitoring & Classification System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: white;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #4ea8de;
        text-align: center;
        margin-bottom: 1rem;
    }

    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #cccccc;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: #f0f0f0;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: left;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }

    .feature-card:hover {
        transform: scale(1.03);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #2563eb, #1e3a8a);
        color: white;
    }

    .feature-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .feature-desc {
        font-size: 0.95rem;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">📊 Log Monitoring & Classification System</h1>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Upload & Classify", "Demo Data", "About"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Upload & Classify":
        show_upload_page()
    elif page == "Demo Data":
        show_demo_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.markdown("## Welcome to the Log Monitoring & Classification System")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h3>🔍 Real-time Classification</h3><p>Upload your log files and get instant classification results using advanced ML models.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h3>🤖 Multiple Models</h3><p>Uses Regex, BERT, and LLM-based classification for accurate results across different log sources.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h3>📈 Analytics Dashboard</h3><p>Visualize classification results with interactive charts and detailed insights.</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Guide")
    st.markdown("""
    1. **Upload your CSV file**
    2. **Classify log messages**
    3. **Download results**
    """)

    st.markdown("## 📋 Supported Log Sources")
    sources = ["ModernCRM", "BillingSystem", "AnalyticsEngine", "ModernHR", "LegacyCRM"]
    for source in sources:
        st.markdown(f"- **{source}**")

def show_upload_page():
    st.markdown("## 📤 Upload & Classify Logs")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### Upload your CSV file (must have 'source' and 'log_message')")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "source" not in df.columns or "log_message" not in df.columns:
                st.error("CSV must contain 'source' and 'log_message' columns.")
                return
            st.success(f"✅ Uploaded {len(df)} entries")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🔍 Start Classification", type="primary"):
                with st.spinner("Classifying logs..."):
                    logs = list(zip(df["source"], df["log_message"]))
                    df["target_label"] = classify(logs)
                    st.success("✅ Classification done")
                    show_classification_results(df)

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_demo_page():
    st.markdown("## 🎯 Demo with Sample Data")
    try:
        demo_df = pd.read_csv("demo.csv")
        st.dataframe(demo_df, use_container_width=True)
        if st.button("🔍 Classify Demo Data", type="primary"):
            with st.spinner("Classifying demo logs..."):
                logs = list(zip(demo_df["source"], demo_df["log_message"]))
                demo_df["target_label"] = classify(logs)
                st.success("✅ Demo classification done")
                show_classification_results(demo_df)
    except Exception as e:
        st.error(f"Error loading demo data: {e}")

def show_classification_results(df):
    st.markdown("## 📊 Classification Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Logs", len(df))
    with col2:
        st.metric("Unique Sources", df["source"].nunique())
    with col3:
        st.metric("Unique Labels", df["target_label"].nunique())
    with col4:
        st.metric("Most Common Label", df["target_label"].mode().iloc[0] if not df["target_label"].mode().empty else "N/A")

    st.markdown("### 📋 Detailed Results")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig_source = px.pie(source_counts, values="count", names="source", title="Log Distribution by Source")
        st.plotly_chart(fig_source, use_container_width=True)
    with col2:
        label_counts = df["target_label"].value_counts().reset_index()
        label_counts.columns = ["target_label", "count"]
        fig_label = px.bar(label_counts, x="target_label", y="count", title="Classification Results")
        st.plotly_chart(fig_label, use_container_width=True)

    st.markdown("### 💾 Download Results")
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="classified_logs.csv", mime="text/csv")

def show_about_page():
    st.markdown("## ℹ️ About This System")
    st.markdown("This tool classifies log messages from various systems using Regex, BERT, and LLM models.")

if __name__ == "__main__":
    main()

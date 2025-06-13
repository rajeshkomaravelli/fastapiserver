import streamlit as st
from modules import Visualization, home, upload, results, report, contact

st.markdown('<link rel="stylesheet" type="text/css" href="assets/styles.css">', unsafe_allow_html=True)

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Promoter Prediction Model",
    layout="wide",
    page_icon="🧬"
)

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "📂 Upload", "📊 Analyze", "📈 Results & Visualization", "📜 Report", "📞 Contact"]
)

# Load the selected page
if page == "🏠 Home":
    home.show()
elif page == "📂 Upload":
    upload.show()
elif page == "📊 Analyze":
    Visualization.show()
elif page == "📈 Results & Visualization":
    results.show()
elif page == "📜 Report":
    report.show()
elif page == "📞 Contact":
    contact.show()

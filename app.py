import streamlit as st
from modules.home import show_home
from modules.upload import show_upload
from modules.Visualization import show_visualization
from modules.results import show_results,show_general_results
from modules.report import show_report
# from modules.contact import show_contact

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Home", "Upload", "Results"],
                        key="navigation_app")

# Load the selected page  
if page == "Home":
    show_home()
elif page == "Upload":
    show_upload()
elif page == "Results":
    if st.session_state.generalized==False:
        show_results()
    else:
        show_general_results()
# elif page == "Visualization":
#     show_visualization()
# elif page == "Report":
#     show_report()
# elif page == "Contact":
#     show_contact()
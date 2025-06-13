import streamlit as st
import os
import glob
from PIL import Image

def show_visualization():
    st.title("Model Performance Visualization")
    if "organism" not in st.session_state:
        st.warning("No organism selected. Please go to the Upload page and submit a sequence first.")
        return

    organism = st.session_state["organism"]
    st.write(f"### Selected Organism: {organism}")
    visualization_folder = f"visualization/{organism}"

    if not os.path.exists(visualization_folder):
        st.error("No visualization data available for the selected organism.")
        return

    st.write("## Model Performance Graphs")
    graph_files = sorted(glob.glob(os.path.join(visualization_folder, "*.png")))

    if not graph_files:
        st.warning("No graphs found in the visualization folder.")
        return

    for graph_file in graph_files:
        image = Image.open(graph_file)
        st.image(image, caption=os.path.basename(graph_file))
if __name__ == "__main__":
    show_visualization()

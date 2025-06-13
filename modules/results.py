import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Add this function to read property names
def load_property_names(file_path):
    """Load property ID to name mapping from the data file"""
    try:
        df = pd.read_csv(file_path, sep='\t| {2,}', engine='python')
        return dict(zip(df['ID'].astype(str), df['PropertyName']))
    except Exception as e:
        st.error(f"Failed to load property names: {str(e)}")
        return {}

# Load property names at startup
PROPERTY_FILE = "C:/Users/saich/Downloads/output (2).txt"
property_names = load_property_names(PROPERTY_FILE)

def show_results():
    """
    Organism-specific results display.
    Uses the /predict endpoint and displays the best property, ensemble prediction,
    model breakdown, and report download.
    """
    st.title("Results & Visualization")
    if "sequence" not in st.session_state or "organism" not in st.session_state:
        st.warning("No sequence submitted. Please go to the Upload page and submit a sequence first.")
        st.stop()

    sequence = st.session_state["sequence"]
    organism = st.session_state["organism"]
    
    st.write(f"### Selected Organism: {organism}")
    st.write(f"### Submitted Sequence: \n`{sequence}`")
    
    FASTAPI_URL = "https://fastapiserver-2.onrender.com/predict"
    with st.spinner("Processing your sequence..."):
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"sequence": sequence, "organism": organism}
            )
            response.raise_for_status()
            results = response.json()
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

    # Get best property name using the mapping
    best_prop_num = results.get('best_property', 'Unknown')
    best_prop_name = property_names.get(str(best_prop_num), f"Unknown Property ({best_prop_num})")

    # Display best property and ensemble prediction
    st.write("## 🔬 Best Property Used")
    st.info(f"**{best_prop_name}**")
    
    st.write("## 🎯 Final Prediction")
    final_pred = results.get('ensemble_prediction', 0)
    prediction_text = "Promoter Sequence ✅" if final_pred == 1 else "Non-Promoter Sequence ❌"
    prediction_color = "green" if final_pred == 1 else "red"
    st.markdown(f"<h2 style='color: {prediction_color};'>{prediction_text}</h2>", unsafe_allow_html=True)
    
    # Display individual model predictions
    st.write("## 📊 Model Predictions Breakdown")
    if isinstance(results.get("predictions"), dict):
        df = pd.DataFrame(
            [(k, "Promoter ✅" if v == 1 else "Non-Promoter ❌") for k, v in results["predictions"].items()],
            columns=["Model", "Prediction"]
        )
        st.session_state["results_df"] = df
        
        # Calculate agreement statistics
        total_models = len(df)
        promoter_votes = sum(df["Prediction"] == "Promoter ✅")
        agreement = promoter_votes / total_models
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Promoter Votes", promoter_votes)
        with col3:
            st.metric("Agreement Rate", f"{agreement:.0%}")

        # Display dataframe with colored predictions
        st.dataframe(
            df.style.applymap(
                lambda x: "color: green" if "Promoter" in x else "color: red",
                subset=["Prediction"]
            ),
            height=400
        )
    else:
        st.error("Invalid predictions format received from server")

    # Download report section
    st.write("## 📥 Download Full Report")
    if "report_path" in results:
        try:
            with open(results["report_path"], "r") as f:
                report_content = f.read()
            
            st.download_button(
                label="Download Prediction Report",
                data=report_content,
                file_name=f"{organism}_promoter_report.txt",
                mime="text/plain"
            )
        except FileNotFoundError:
            st.warning("Could not find the report file")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    else:
        st.warning("No report available for download")

def show_general_results():
    """
    Generalized results display.
    Uses the /predict_regions endpoint to obtain:
      {
          "organism": organism,
          "input_sequence_length": seq_length,
          "best_precision": best_precision,
          "best_recall": best_recall,
          "score_profile": score_profile,
          "promoter_regions": promoter_regions
      }
    The function then highlights the predicted promoter regions (using a yellow background)
    within the input sequence and displays the overall prediction details.
    """
    st.title("Generalized Results & Visualization")
    if "sequence" not in st.session_state or "organism" not in st.session_state:
        st.warning("No sequence submitted for the generalized model. Please go to the Upload page and submit a sequence first.")
        st.stop()

    sequence = st.session_state["sequence"]
    organism = st.session_state["organism"]
    
    st.write(f"### Selected Organism: {organism}")
    st.write(f"### Submitted Sequence: \n`{sequence}`")
    
    FASTAPI_GENERAL_URL = "http://127.0.0.1:8000/predict_regions"
    with st.spinner("Processing your sequence for potential promoter regions..."):
        try:
            response = requests.post(
                FASTAPI_GENERAL_URL,
                json={"sequence": sequence, "organism": organism}
            )
            response.raise_for_status()
            results = response.json()
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

    # Display overall output details
    st.write("## General Prediction Output")
    # st.write(f"**Organism:** {results.get('organism', 'Unknown')}")
    st.write(f"**Input Sequence Length:** {results.get('input_sequence_length', 'Unknown')}")
    st.write(f"**Best Precision:** {results.get('best_precision', 'Unknown')}")
    st.write(f"**Best Recall:** {results.get('best_recall', 'Unknown')}")
    st.write("**Score Profile:**")
    st.write(results.get('score_profile', []))
    
    # Process and highlight promoter regions from the score profile
    promoter_regions = results.get("promoter_regions", [])
    if promoter_regions:
        st.write("## Potential Promoter Regions")
        # Create an array to mark indices with potential promoters
        highlight_indices = [False] * len(sequence)
        for region in promoter_regions:
            region_start = region.get("start")
            region_end = region.get("end")
            for i in range(region_start, region_end + 1):
                highlight_indices[i] = True

        # Build HTML output that highlights the promoter region characters in yellow
        html_sequence = ""
        for i, char in enumerate(sequence):
            if highlight_indices[i]:
                html_sequence += f"<span style='background-color: red;'>{char}</span>"
            else:
                html_sequence += char
                
        st.write("### Highlighted Sequence (Promoter Regions in Red)")
        st.markdown(html_sequence, unsafe_allow_html=True)
        
        st.write("#### Detailed Promoter Regions:")
        for region in promoter_regions:
            st.write(f"**Range:** {region.get('start')} - {region.get('end')}")
            st.write(f"**Region Sequence:** `{region.get('region_sequence')}`")
    else:
        st.write("## No potential promoter regions were identified.")

# Main execution: choose which result display function to run.
if __name__ == "__main__":
    # Use st.session_state.generalized to determine which results function to call.
    # Default to organism-specific results if generalized is not set or is False.
    # if st.session_state.get("generalized", True):
    #     show_results()
    # else:
    show_general_results()

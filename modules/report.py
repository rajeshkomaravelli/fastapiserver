import os
import streamlit as st
import pandas as pd
from fpdf import FPDF

def create_pdf(results_df, organism, image_paths):
    """Generates a PDF report with results table and graphs."""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"Promoter Prediction Report - {organism}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    headers = list(results_df.columns)
    col_widths = [40] * len(headers) 
    for header in headers:
        pdf.cell(col_widths[headers.index(header)], 10, header, border=1, align="C")
    pdf.ln()
    pdf.set_font("Arial", "", 10)
    for _, row in results_df.iterrows():
        for header in headers:
            pdf.cell(col_widths[headers.index(header)], 10, str(row.get(header, "-")), border=1, align="C")
        pdf.ln()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Performance Metrics Graphs", ln=True, align="C")
    pdf.ln(5)
    for image_path in image_paths:
        if os.path.exists(image_path):
            pdf.image(image_path, x=10, w=180)  
            pdf.ln(10)
    pdf_path = f"{organism}_report.pdf"
    pdf.output(pdf_path)
    
    return pdf_path

def show_report():
    """Streamlit function to generate and download the report."""
    
    st.title("Download Report")
    if "results_df" not in st.session_state or "organism" not in st.session_state:
        st.warning("No results available. Please analyze the sequences first.")
        return

    results_df = st.session_state["results_df"]
    organism = st.session_state["organism"]
    if results_df.empty:
        st.warning("No data available to generate the report.")
        return
    visualization_dir = f"visualization/{organism}"
    image_paths = [os.path.join(visualization_dir, img) for img in os.listdir(visualization_dir) if img.endswith(".png")]

    if not image_paths:
        st.warning("No visualization graphs found for this organism.")
    pdf_path = create_pdf(results_df, organism, image_paths)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_file,
            file_name=f"{organism}_report.pdf",
            mime="application/pdf"
        )

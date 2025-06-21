import streamlit as st 
from Bio import SeqIO
import io

def validate_sequence_general(sequence):
    """Check if sequence is at least 150 bases and contains only A, T, G, C."""
    sequence = sequence.strip().upper()
    if len(sequence) < 150:
        return "The sequence must be at least 150 bases long."
    elif any(base not in "ATGC" for base in sequence):
        return "The sequence must only contain A, T, G, or C."
    return None

def validate_sequence_exact(sequence):
    """Check if the sequence is exactly 150 bases and contains only A, T, G, C."""
    sequence = sequence.strip().upper()
    if len(sequence) != 150:
        return "The sequence must be exactly 150 bases long."
    elif any(base not in "ATGC" for base in sequence):
        return "The sequence must only contain A, T, G, or C."
    return None

def extract_fasta_sequence(fasta_file, min_length=150, exact_length=True):
    """Extracts sequences from a FASTA file and ensures they meet the criteria."""
    try:
        fasta_content = fasta_file.read().decode("utf-8")
        fasta_io = io.StringIO(fasta_content)
        sequences = [str(record.seq).strip().upper() for record in SeqIO.parse(fasta_io, "fasta")]
        if not sequences:
            return None, "The FASTA file does not contain any valid sequences."
        for seq in sequences:
            if exact_length:
                error = validate_sequence_exact(seq)
            else:
                error = validate_sequence_general(seq)
            if error:
                return None, error
        return sequences[0], None
    except Exception as e:
        return None, f"Error processing FASTA file: {e}"

# -- Callback functions to update session state --
def update_specific_sequence():
    # When user types in the organism-specific text area, update session_state.
    if st.session_state.sequence_input:
        st.session_state.sequence = st.session_state.sequence_input.strip().upper()

def update_general_sequence():
    # When user types in the general model text area, update session_state.
    if st.session_state.general_sequence_input:
        st.session_state.sequence = st.session_state.general_sequence_input.strip().upper()

def update_specific_file():
    # When a FASTA file is uploaded for the specific model, update st.session_state.sequence.
    if st.session_state.fasta_upload is not None:
        seq, err = extract_fasta_sequence(st.session_state.fasta_upload)
        if seq:
            st.session_state.sequence = seq.strip().upper()

def update_general_file():
    # When a FASTA file is uploaded for the general model, update st.session_state.sequence.
    if st.session_state.general_fasta_upload is not None:
        seq, err = extract_fasta_sequence(st.session_state.general_fasta_upload, min_length=150, exact_length=False)
        if seq:
            st.session_state.sequence = seq.strip().upper()

def show_upload():
    st.title("Upload Sequence for Promoter Prediction")

    st.write("""
        ## Instructions
        - Submit either a **nucleotide sequence** (150 bases) or a **FASTA file**.
        - The sequence should be in **A, T, G, C** format (no special characters).
        - If uploading a FASTA file, ensure all sequences are exactly **150 bases** long.
    """)

    input_option = st.radio("Choose input method:", ("Submit Sequence", "Upload FASTA File"), key="input_method")

    # Initialize session state variables if not set.
    if "sequence" not in st.session_state:
        st.session_state.sequence = ""
    if "organism" not in st.session_state:
        st.session_state.organism = ""
    if "generalized" not in st.session_state:
        st.session_state.generalized = False

    if input_option == "Submit Sequence":
        st.text_area("Enter your 150-length nucleotide sequence:", height=150, key="sequence_input", on_change=update_specific_sequence)
        if st.session_state.sequence:
            st.caption(f"Length: {len(st.session_state.sequence)} bases")
    else:
        st.file_uploader("Upload FASTA File:", type=["fasta", "fa", "txt"], key="fasta_upload", on_change=update_specific_file)
        if st.session_state.sequence:
            st.caption(f"Length: {len(st.session_state.sequence)} bases")

    # organisms = [
    #     "Haloferax_volcanii_DS2", "Helicobacter pylori 26695", "Klebsiella pneumoniae subsp. pneumoniae MGH 78578",
    #     "Mycobacterium tuberculosis H37Rv", "Nostoc sp. PCC 7120 = FACHB-418", "Pseudomonas aeruginosa UCBPP-PA14",
    #     "Saccharolobus solfataricus P2", "Salmonella enterica subsp. enterica serovar Typhimurium str. SL1344",
    #     "Streptomyces coelicolor A3(2)", "Synechocystis sp. PCC 6803", "Thermococcus kodakarensis KOD1",
    #     "Bacillus amyloliquefaciens XH7", "Chlamydia pneumoniae CWL029", "Corynebacterium glutamicum ATCC 13032",
    #     "Escherichia coli str. K-12 substr. MG1655"
    # ]
    # selected_organism = st.selectbox("Select the organism:", ["Select an organism"] + organisms, key="organism_selection")

    if st.button("Submit", key="submit_button"):
        errors = []
        # if selected_organism == "Select an organism":
        #     errors.append("Please select an organism.")
        if input_option == "Submit Sequence":
            seq_error = validate_sequence_exact(st.session_state.sequence)
            if seq_error:
                errors.append(seq_error)
        else:
            if st.session_state.get("fasta_upload") is None:
                errors.append("Please upload a FASTA file.")
        if errors:
            for err in errors:
                st.warning(err)
        else:
            st.session_state.organism = "allorganism"
            # For organism-specific submission, mark as not generalized.
            st.session_state.generalized = False
            st.success("Sequence submitted successfully! Please go to the results page to see the results.")

    # st.markdown("---")
#     st.header("Generalized Bacterial Promoter Prediction (Any Bacterial Organism)")
#     st.write("""
#         Using data from 15 different bacterial organisms, we have trained a **generalized promoter prediction model**.
#         This model can determine whether a given nucleotide sequence (from **any bacterial genome**) is likely to be a **promoter**.
    
#         ### Requirements:
#         - Sequence must be **at least 150 bases** long.
#         - Only characters **A, T, G, C** are allowed.
#         - Input via **text** or **FASTA file** is accepted.
#     """)

#     general_input_option = st.radio("Choose input method for general model:", ("Submit Sequence", "Upload FASTA File"), key="general_input")

#     if general_input_option == "Submit Sequence":
#         st.text_area("Enter your nucleotide sequence (≥150 bases):", height=150, key="general_sequence_input", on_change=update_general_sequence)
#         if st.session_state.sequence:
#             st.caption(f"Length: {len(st.session_state.sequence)} bases")
#     else:
#         st.file_uploader("Upload FASTA File:", type=["fasta", "fa", "txt"], key="general_fasta_upload", on_change=update_general_file)
#         if st.session_state.sequence:
#             st.caption(f"Length: {len(st.session_state.sequence)} bases")

#     if st.button("Submit to General Model", key="general_submit_button"):
#         errors = []
#         if general_input_option == "Submit Sequence":
#             general_error = validate_sequence_general(st.session_state.sequence)
#             if general_error:
#                 errors.append(general_error)
#         else:
#             if st.session_state.get("general_fasta_upload") is None:
#                 errors.append("Please upload a FASTA file.")
#         if errors:
#             for err in errors:
#                 st.warning(err)
#         else:
#             # For general model submission, you may optionally override the organism.
#             st.session_state.organism = "allorganism"
#             st.session_state.generalized = True
#             st.success("Sequence submitted to generalized model successfully! Please go to the results page to view prediction.")

# if __name__ == "__main__":
#     show_upload()
#     print(st.session_state.generalized)

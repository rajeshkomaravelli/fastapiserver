import streamlit as st
# Sidebar Navigation
def show_home():
    st.title("Promoter Prediction Model")
    st.markdown(
        """
        ### About
        This application uses machine learning models to predict promoter sequences for various organisms.
        The models have been trained using **kappa encoding** on promoter region sequences of length 150.
        Below is the list of supported organisms.
        Click on an organism name to learn more about it.
        """
    )
    
    organisms = {
        "Haloferax volcanii DS2": "https://en.wikipedia.org/wiki/Haloferax_volcanii",
        "Helicobacter pylori 26695": "https://en.wikipedia.org/wiki/Helicobacter_pylori",
        "Klebsiella pneumoniae MGH 78578": "https://en.wikipedia.org/wiki/Klebsiella_pneumoniae",
        "Mycobacterium tuberculosis H37Rv": "https://en.wikipedia.org/wiki/Mycobacterium_tuberculosis",
        "Nostoc sp. PCC 7120": "https://en.wikipedia.org/wiki/Nostoc",
        "Pseudomonas aeruginosa UCBPP-PA14": "https://en.wikipedia.org/wiki/Pseudomonas_aeruginosa",
        "Saccharolobus solfataricus P2": "https://en.wikipedia.org/wiki/Sulfolobus_solfataricus",
        "Salmonella enterica SL1344": "https://en.wikipedia.org/wiki/Salmonella",
        "Streptomyces coelicolor A3(2)": "https://en.wikipedia.org/wiki/Streptomyces_coelicolor",
        "Synechocystis sp. PCC 6803": "https://en.wikipedia.org/wiki/Synechocystis",
        "Thermococcus kodakarensis KOD1": "https://en.wikipedia.org/wiki/Thermococcus_kodakarensis",
        "Bacillus amyloliquefaciens XH7": "https://en.wikipedia.org/wiki/Bacillus_amyloliquefaciens",
        "Chlamydia pneumoniae CWL029": "https://en.wikipedia.org/wiki/Chlamydia_pneumoniae",
        "Corynebacterium glutamicum ATCC 13032": "https://en.wikipedia.org/wiki/Corynebacterium_glutamicum",
        "Escherichia coli MG1655": "https://en.wikipedia.org/wiki/Escherichia_coli"
    }
    
    for name, url in organisms.items():
        st.link_button(name, url)   
    
    st.markdown(
        """
        ### Machine Learning Models
        The following ML algorithms are used for promoter sequence prediction:
        - Support Vector Machine (SVM)
        - Random Forest
        - Logistic Regression
        - Naive Bayes
        - K-Nearest Neighbors (K-NN)
        - Gradient Boosting
        - AdaBoost
        - Decision Tree
        - Perceptron
        - Stochastic Gradient Descent (SGD)
        - Bagging Classifier
        - Extra Trees Classifier
        - CatBoost
        - LightGBM
        - XGBoost
        
        Each of these models has been trained on promoter sequences for each of the listed organisms, and the trained models have been saved as pickle files.
        """
    )


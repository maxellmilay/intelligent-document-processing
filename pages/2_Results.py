import streamlit as st
import pandas as pd

st.title("Processing Results")

st.write("This page displays the results of document processing")

# Sample data - in a real app, this would come from your processing pipeline
sample_data = {
    "Document": ["invoice.pdf", "contract.docx", "report.txt"],
    "Status": ["Processed", "In Progress", "Processed"],
    "Confidence": [0.92, 0.78, 0.85]
}

df = pd.DataFrame(sample_data)
st.dataframe(df) 
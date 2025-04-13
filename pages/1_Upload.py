import streamlit as st

st.title("Document Upload")

st.write("This is the upload page where users can upload documents for processing")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    # Here you would process the file with your existing document processing logic
    st.write("Processing the document...") 
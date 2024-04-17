import streamlit as st
import pdfplumber
import textract
import os
import io
from docx import Document
from data_cleaning_module import clean_data
from cosine_similarity_module import calculate_cosine_similarity
from tfidf_module import identify_important_parts
from topic_modeling_module import model_topics
from clustering_module import cluster_data
from summarization_module import *

# Custom CSS styles
custom_styles = """
<style>
body {
    background-color: #f0f2f6;
    color: #333;
    font-family: Arial, sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stTextInput>div>div>input {
    border-radius: 4px;
    border-color: #ccc;
}
.stTextInput>div>div>input:focus {
    border-color: #4CAF50;
    box-shadow: none;
}
.stFileUploader>div>div>label {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
}
.stFileUploader>div>div>label:hover {
    background-color: #45a049;
}
.stProgress>div>div>div {
    background-color: #4CAF50;
}
.stProgress>div>div>div>div {
    background-color: #45a049;
}
.stProgress>div>div>div>div>div {
    background-color: #4CAF50;
}
</style>
"""

# Function to read text from PDF, DOC, and TXT files
def read_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith((".doc", ".docx")):
        text = textract.process(file_path).decode("utf-8")
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            text = file.read()
        return text

# Function to dynamically calculate threshold
def calculate_threshold(data):
    total_docs = len(data)
    total_text_length = sum(len(doc) for doc in data)

    # Adjust thresholds based on your specific criteria
    if total_docs > 10 or total_text_length > 10000:
        threshold = 10  # Advanced processing
    else:
        threshold = 5  # Basic processing

    return threshold

# Function to determine file type
def get_file_extension(file_name):
    _, extension = os.path.splitext(file_name)
    return extension

# Function to read text from DOCX files
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to upload and process documents
def main():
    st.title("Document Summarization App")
    st.markdown(custom_styles, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload your files", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        st.success("Files uploaded successfully!")
        if st.button("Summarize All Documents"):
            summaries = []
            with st.spinner('Processing documents...'):
                for idx, file in enumerate(uploaded_files):
                    st.header(f"Document {idx + 1}")
                    file_content = file.read()
                    file_extension = os.path.splitext(file.name)[1]

                    if file_extension == ".pdf":
                        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                            text = ""
                            for page in pdf.pages:
                                text += page.extract_text()
                    elif file_extension == ".docx":
                        text = read_docx(io.BytesIO(file_content))  # Using python-docx for .docx files
                    elif file_extension == ".doc":
                        text = read_docx(file)  # Using python-docx for .doc files
                    else:  # For .txt files
                        text = file_content.decode("utf-8")

                    #---------------------CLEANING----------------#
                    cleaned_data = clean_data([text])

                    #---------------------COSINE SIMILARITY----------------#
                    cosine_data = calculate_cosine_similarity(cleaned_data)

                    #---------------------TFIDF SIMILARITY----------------#
                    important_parts = identify_important_parts(cosine_data)

                    #---------------------TOPIC MODELLING----------------#
                    topic_insights = model_topics(important_parts)

                    #---------------------CLUSTERING ----------------#
                    clustered_data = cluster_data(important_parts)

                    #---------------------SUMMARY ----------------#
                    st.write(f"Generating summary for Document {idx + 1}...")
                    summary = summarize_text_with_openai(clustered_data)

                    # Store the summary
                    summaries.append((f"Document {idx + 1}", summary))

                    # Display the summary
                    st.subheader(f"Summary for Document {idx + 1}:")
                    st.write(summary)

            # Display summaries for all documents
            if summaries:
                st.header("Summaries for All Documents")
                all_summaries = ""
                for idx, summary_tuple in enumerate(summaries):
                    doc_num, summary = summary_tuple
                    all_summaries += f"Document {doc_num} Summary:\n{summary}\n\n"
                st.write(all_summaries)

if __name__ == '__main__':
    main()

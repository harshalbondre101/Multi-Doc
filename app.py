import streamlit as st
import pdfplumber
import os
import io
from docx import Document
from data_cleaning_module import clean_data
from cosine_similarity_module import calculate_cosine_similarity
from tfidf_module import identify_important_parts
from topic_modeling_module import model_topics
from clustering_module import cluster_data
from summarization_module import *


# Function to read text from PDF, DOC, and TXT files
def read_text(file_content, file_extension):
    if file_extension == ".pdf":
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    elif file_extension == ".docx":
        text = read_docx(io.BytesIO(file_content))
        return text
    elif file_extension == ".txt":
        text = file_content.decode("utf-8")
        return text
    else:
        return None
    
# Function to read text from DOCX files
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

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

# Function to upload and process documents
def main():
    st.title("Document Summarization App")

    uploaded_files = st.file_uploader("Upload your files", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)
    summaries = []
    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            st.header(f"Document {idx + 1}")
            file_content = file.read()
            file_extension = os.path.splitext(file.name)[1]

            text = read_text(io.BytesIO(file_content))

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
            summary =  generate_summary(clustered_data)

            # Store the summary
            summaries.append((f"Document {idx + 1}", summary))

            # Display the summary
            #st.subheader(f"Summary for Document {idx + 1}:")
            #st.write(summary)
    
    # Button to display summaries for all documents

    if summaries:
        if st.button("View Summaries of all documents:"):
            st.header("Summaries for All Documents")
            all_summaries = ""
            for idx, summary_tuple in enumerate(summaries):
                doc_num, summary = summary_tuple
                all_summaries += f"Document {doc_num} Summary:\n{summary}\n\n"
            st.write(all_summaries)


# Run the app
if __name__ == '__main__':
    main()

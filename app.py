import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Set device for BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Function to get BERT embeddings for a text input
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu()

# Function to safely decode the file content with fallback encoding
def decode_file(file):
    try:
        return file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return file.getvalue().decode("ISO-8859-1")

# Streamlit App Interface
st.title("Automated Resume Screening")

# Upload Job Description
jd_file = st.file_uploader("Upload Job Description (.txt)", type="txt")
# Upload Resumes (multiple files allowed)
resume_files = st.file_uploader("Upload Resumes (.txt)", type="txt", accept_multiple_files=True)

# Process files and calculate similarity
if jd_file and resume_files:
    # Read and preprocess job description
    jd_content = decode_file(jd_file)  # Use the decode_file function
    jd_content = preprocess_text(jd_content)
    jd_embedding = get_bert_embeddings(jd_content)

    # Process each resume and calculate similarity
    results = []
    for resume_file in resume_files:
        # Read and preprocess resume
        resume_content = decode_file(resume_file)  # Use the decode_file function
        resume_content = preprocess_text(resume_content)
        resume_embedding = get_bert_embeddings(resume_content)

        # Calculate similarity score
        similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
        results.append({
            'Job Description': jd_file.name,
            'Resume': resume_file.name,
            'Similarity Score': similarity_score
        })

    # Display results in a DataFrame
    results_df = pd.DataFrame(results).sort_values(by='Similarity Score', ascending=False)
    st.write("### Similarity Scores")
    st.dataframe(results_df)
else:
    st.write("Please upload a job description and at least one resume to get similarity scores.")

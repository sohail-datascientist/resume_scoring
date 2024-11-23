import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from groq import Groq
import json
import fitz  # PyMuPDF for PDF processing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set device for BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

###################### Start #######################
# Llama 3.1 Initialization
client = Groq(api_key="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")

# Adjusted prompt for JSON output
instruction = """
You are an AI bot designed to parse resumes and extract the following details in JSON:
1. full_name
2. university_name
3. national_university/international_university "if national return Yes else No"
4. email_id
5. employment_details (with fields: company, position, duration, location, and tags indicating teaching, industry, or internship based on role)
6. technical_skills
7. soft_skills
8. location

Classify university as either "National University" or "International University".
Only return the most relevant job roles without categorizing experience.
Return all information in JSON format, including the roles tagged with the correct university type and job details.
"""
###################### End #######################

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

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Custom CSS Styling for the Streamlit App
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: bold;
        color: #3498db;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stDataFrame {
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stMarkdown {
        margin-top: 20px;
    }
    table, th, td {
        border: 1px solid #ccc;
    }
    th, td {
        padding: 8px 12px;
    }
    .stAlert {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Set up Streamlit app UI
st.title("Automated Resume Screening Dashboard")

# Sidebar - File Upload
with st.sidebar:
    st.header("📤 File Uploads")
    jd_file = st.file_uploader("Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])
    resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

########################## Main Body ###########################

# Only process if JD and resumes are uploaded
if jd_file and resume_files:
    # Initialize results_df only after resumes are processed
    results_df = pd.DataFrame(columns=["Resume", "Similarity Score", "full_name", "university_name", "company_names", "technical_skills", "soft_skills", "experience"])

    # Process files and calculate similarity only if resumes are uploaded
    if jd_file and resume_files:
        if jd_file.type == "application/pdf":
            jd_content = extract_text_from_pdf(jd_file)
        else:
            jd_content = decode_file(jd_file)
        jd_content = preprocess_text(jd_content)
        jd_embedding = get_bert_embeddings(jd_content)

        results = []
        for resume_file in resume_files:
            if resume_file.type == "application/pdf":
                resume_content = extract_text_from_pdf(resume_file)
            else:
                resume_content = decode_file(resume_file)
            resume_content = preprocess_text(resume_content)
            resume_embedding = get_bert_embeddings(resume_content)
            similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]

            # Request data extraction from Groq
            completion = client.chat.completions.create(
                model="llama3-groq-70b-8192-tool-use-preview",
                messages=[{"role": "user", "content": instruction + resume_content}],
                temperature=0.5, max_tokens=1024, top_p=0.65
            )

            try:
                result_json = completion.choices[0].message.content
                result = json.loads(result_json)
            except json.JSONDecodeError:
                result = {}

            employment_details = result.get("employment_details", [])
            
            # Store employment details and classify
            if not employment_details:
                experience = "Fresh Candidate"
                company_names = ["N/A"]
                skills = result.get("technical_skills", [])
            else:
                experience = "Experienced"
                company_names = [detail.get('company', 'N/A') for detail in employment_details]
                skills = result.get("technical_skills", [])

            # Append data to results
            results.append({
                'Resume': resume_file.name,
                'Similarity Score': similarity_score,
                'full_name': result.get("full_name"),
                'university_name': result.get("university_name"),
                'company_names': company_names,
                'technical_skills': skills,
                'soft_skills': " - ".join(result.get("soft_skills", [])),
                'experience': experience
            })

        results_df = pd.DataFrame(results)

    # Sort results by Similarity Score in descending order
    results_df = results_df.sort_values(by="Similarity Score", ascending=False)
    # Display filtered table
    st.write("### Candidates")
    st.dataframe(results_df)

    ######################### Filter Section ######################
    st.write("### Apply Filters")
    
    # Filters for universities and companies
    universities = st.multiselect("Select Universities", options=results_df["university_name"].unique())
    companies = st.multiselect("Select Companies", options=results_df["company_names"].explode().unique())
    skills = st.multiselect("Select Skills", options=results_df['technical_skills'].explode().unique())
    
    # Filter results based on selections
    filtered_df = results_df.copy()
    
    # Filter by University
    if universities:
        filtered_df = filtered_df[filtered_df["university_name"].isin(universities)]
        st.write("### Filtered Candidates (By University)")
        st.dataframe(filtered_df)
    
    # Filter by Company
    if companies:
        filtered_df = filtered_df[filtered_df['company_names'].apply(lambda x: any(company in companies for company in x))]
        st.write("### Filtered Candidates (By Company)")
        st.dataframe(filtered_df)
    
    # Filter by Skills (if applicable)
    if skills:
        filtered_df = filtered_df[filtered_df['technical_skills'].apply(lambda x: any(skill in skills for skill in x))]
        st.write("### Filtered Candidates (By Skills)")
        st.dataframe(filtered_df)
    
    ######################### Resume Statistics Table ######################
    # Experience and university/company counts
    flattened_company_names = [company for sublist in filtered_df['company_names'] for company in sublist]
    unique_companies = list(set(flattened_company_names))

    experience_counts = {
        "Fresh Candidate": 0,
        "Experienced": 0
    }
    for experience in filtered_df['experience']:
        experience_counts[experience] += 1

    # Resume Statistics as a Table
    resume_stats = pd.DataFrame({
        "Total Resumes": [len(results_df)],
        "Fresh Candidates": [experience_counts['Fresh Candidate']],
        "Experienced Candidates": [experience_counts['Experienced']],
    })

    st.write("### Resume Statistics")
    st.dataframe(resume_stats)

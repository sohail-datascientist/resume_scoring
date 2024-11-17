import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from groq import Groq
import json
import fitz  # PyMuPDF for PDF extraction

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
You are an AI bot designed to act as a professional for parsing resumes. 
You are given a resume, and your job is to extract the following information in JSON format only:
    1. full_name
    2. university_name
    3. national/international_university "if national return Yes else No"
    4. email_id
    5. employment_details (with fields: company, position, duration, location)
    6. technical_skills
    7. soft_skills
    8. location "Please provide the location of the candidate's work experience or location mentioned in the resume"
Give the extracted information in JSON format only, without any additional commentary.
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

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the file directly as a BytesIO object
    pdf_bytes = pdf_file.read()  # Read the BytesIO content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Use fitz.open() with stream argument
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text

# Streamlit App Interface
st.title("Automated Resume Screening")

# Upload Job Description
jd_file = st.file_uploader("Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])

# Upload Resumes (multiple files allowed)
resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

# Initialize variables for filtered data
filtered_results = pd.DataFrame()

# Process files and calculate similarity
if jd_file and resume_files:
    # Handle Job Description file upload (TXT or PDF)
    if jd_file.type == "application/pdf":
        jd_content = extract_text_from_pdf(jd_file)
    else:
        jd_content = decode_file(jd_file)
    
    jd_content = preprocess_text(jd_content)
    jd_embedding = get_bert_embeddings(jd_content)

    progress_bar = st.progress(0)
    total_files = len(resume_files)
    results = []

    for idx, resume_file in enumerate(resume_files):
        progress_bar.progress((idx + 1) / total_files)
        
        # Handle Resume file upload (TXT or PDF)
        if resume_file.type == "application/pdf":
            resume_content = extract_text_from_pdf(resume_file)
        else:
            resume_content = decode_file(resume_file)
        
        resume_content = preprocess_text(resume_content)
        resume_embedding = get_bert_embeddings(resume_content)

        similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]

        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[{"role": "user", "content": instruction + resume_content}],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=False,
        )

        result_json = completion.choices[0].message.content

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            result = {}

        employment_details = result.get("employment_details", [])
        if not employment_details:
            employment_details = [{'company': 'Fresh Candidate', 'position': 'Fresh Candidate', 'duration': 'N/A', 'location': 'N/A'}]

        company_names = []
        locations = []
        designations = []

        for detail in employment_details:
            company_name = detail.get('company', 'N/A')
            designation = detail.get('position', 'N/A')
            location = detail.get('location', 'N/A')
            if location == 'N/A':
                location = result.get('location', 'N/A')  # Directly get location from the result

            company_names.append(company_name)
            locations.append(location)
            designations.append(designation)

        company_names_str = " - ".join(company_names)
        location_str = locations[0] if locations else 'N/A'
        designation_str = ", ".join(designations)

        technical_skills = result.get("technical_skills", [])
        soft_skills = result.get("soft_skills", [])
        technical_skills_str = " - ".join(technical_skills)
        soft_skills_str = " - ".join(soft_skills)

        results.append({
            'Job Description': jd_file.name,
            'Resume': resume_file.name,
            'Similarity Score': similarity_score,
            'full_name': result.get("full_name"),
            'university_name': result.get("university_name"),
            'national/international_university': result.get("national/international_university"),
            'email_id': result.get("email_id"),
            'technical_skills': technical_skills_str,
            'soft_skills': soft_skills_str,
            'company_names': company_names_str,
            'location': location_str,
            'designation': designation_str,
        })

    results_df = pd.DataFrame(results).sort_values(by='Similarity Score', ascending=False)

    st.write("### Original Results")
    st.dataframe(results_df)

    university_names = results_df['university_name'].dropna().unique()
    selected_universities = st.multiselect('Filter by Universities', university_names)

    all_company_names = results_df['company_names'].str.split(' - ').explode().dropna().unique()
    selected_companies = st.multiselect('Filter by Companies', all_company_names)

    all_skills = results_df['technical_skills'].str.split(' - ').explode().dropna().unique()
    selected_skills = st.multiselect('Filter by Skills', all_skills)

    filtered_results = results_df.copy()

    if selected_universities:
        filtered_results = filtered_results[filtered_results['university_name'].isin(selected_universities)]

    if selected_companies:
        filtered_results = filtered_results[filtered_results['company_names'].str.contains('|'.join(selected_companies), na=False)]

    # if selected_skills:
    #     filtered_results = filtered_results[filtered_results['technical_skills'].str.contains('|'.join(selected_skills), na=False)]

    # if not filtered_results.empty:
        st.write(f"### Filtered Results ({len(filtered_results)} candidates)")
        st.dataframe(filtered_results)
    else:
        st.write("No candidates match the selected criteria.")

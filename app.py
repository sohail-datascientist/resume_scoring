import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from groq import Groq
import json
import fitz
import matplotlib.pyplot as plt
import seaborn as sns

# Set device for BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

###################### API Client Setup #######################
api_keys = [
    "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",  # Client 1
    "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",  # Client 2
    "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"     # Client 3
]

clients = [Groq(api_key=key) for key in api_keys if key.strip()]

instruction = """
You are an AI bot designed to parse resumes and extract the following details in below JSON:
1. full_name: 
2. university_name: of most recent degree (return the short form of the university name else return full name) 
3. national_university/international_university: "return National if inside Pak else return International" 
4. email_id: if available else return "N/A"
5. github_link: if available else return "N/A"
6. employment_details: (company, position, years_of_experience, location, tags: teaching/industry/internship)
7. total_professional_experience: total experience in years excluding internships (return Fresh Graduate if not available)
8. technical_skills:
9. soft_skills: 
10. location: 

Return all information in JSON format.
"""
###################### Core Functions #######################

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu()

def decode_file(file):
    try:
        return file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return file.getvalue().decode("ISO-8859-1")

def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

###################### Streamlit UI #######################
st.markdown("""
    <style>
    [Keep the original CSS styles here]
    </style>
""", unsafe_allow_html=True)

st.title("Automated Resume Screening Dashboard")

# Sidebar - File Upload
with st.sidebar:
    st.header("📤 File Uploads")
    jd_file = st.file_uploader("Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])
    resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

###################### Main Processing #######################
if jd_file and resume_files:
    results_df = pd.DataFrame(columns=["Resume", "Similarity Score", "full_name", "university_name",
                                      "national/international  uni.","email_id","github_link", "company_names",
                                      "technical_skills", "soft_skills", "Total experience in Years", "location"])

    # Process JD first
    if jd_file.type == "application/pdf":
        jd_content = extract_text_from_pdf(jd_file)
    else:
        jd_content = decode_file(jd_file)
    jd_content = preprocess_text(jd_content)
    jd_embedding = get_bert_embeddings(jd_content)

    # Process resumes
    for resume_file in resume_files:
        if resume_file.type == "application/pdf":
            resume_content = extract_text_from_pdf(resume_file)
        else:
            resume_content = decode_file(resume_file)
        
        processed_content = preprocess_text(resume_content)
        resume_embedding = get_bert_embeddings(processed_content)
        similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]

        # API Call with Client Rotation
        completion = None
        used_client = None
        
        for i, client in enumerate(clients):
            try:
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": instruction + processed_content}],
                    temperature=0,
                    max_tokens=1024,
                    top_p=0.65,
                    response_format={"type": "json_object"}
                )
                used_client = i+1
                break  # Successful call
            except Exception as e:
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    st.warning(f"Client {i+1} limit exceeded. Trying next...")
                    continue
                else:
                    st.error(f"Error with Client {i+1}: {str(e)}")
                    break

        if not completion:
            st.error(f"Skipped resume due to API limits: {resume_file.name}")
            continue

        # Process API response
        try:
            result = json.loads(completion.choices[0].message.content)
            employment_details = result.get("employment_details", [])
            company_names = [detail.get('company', 'N/A') for detail in employment_details] if employment_details else ["N/A"]
            
            results_df.loc[len(results_df)] = {
                "Resume": resume_file.name,
                "Similarity Score": similarity_score,
                "full_name": result.get("full_name", "N/A"),
                "university_name": result.get("university_name", "N/A"),
                "national/international  uni.": result.get("national_university/international_university", "N/A"),
                "email_id": result.get("email_id", "N/A"),
                "github_link": result.get("github_link", "N/A"),
                "company_names": company_names,
                "technical_skills": result.get("technical_skills", []),
                "soft_skills": result.get("soft_skills", []),
                "Total experience in Years": result.get("total_professional_experience", "N/A"),
                "location": result.get("location", "N/A")
            }
            
            st.success(f"Processed {resume_file.name} using Client {used_client}")

        except json.JSONDecodeError:
            st.error(f"Failed to parse response for: {resume_file.name}")
        except Exception as e:
            st.error(f"Error processing {resume_file.name}: {str(e)}")

    # Sort and display results
    results_df = results_df.sort_values(by="Similarity Score", ascending=False)
    st.write("### Candidates Ranking")
    st.dataframe(results_df)

    ###################### Visualization & Filtering #######################
    st.write("### Advanced Analytics")
    
    # University Distribution
    uni_counts = results_df['university_name'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x=uni_counts.values, y=uni_counts.index, palette="viridis")
    plt.title("University Distribution")
    st.pyplot(fig1)

    # Experience Distribution
    fig2, ax2 = plt.subplots(figsize=(8,6))
    results_df['Total experience in Years'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.ylabel("")
    st.pyplot(fig2)

    # Skill Word Cloud
    all_skills = [skill for sublist in results_df['technical_skills'] for skill in sublist]
    if all_skills:
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_skills))
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis("off")
        st.pyplot(fig3)

    ###################### Interactive Filters #######################
    st.write("### Candidate Filtering")
    
    col1, col2 = st.columns(2)
    with col1:
        min_exp = st.slider("Minimum Experience (Years)", 
                          min_value=0.0,
                          max_value=float(results_df['Total experience in Years'].max()),
                          value=0.0)
        
    with col2:
        skill_filter = st.multiselect("Technical Skills", 
                                    options=list(set(all_skills)))

    filtered_df = results_df[
        (results_df['Total experience in Years'] >= min_exp) &
        (results_df['technical_skills'].apply(
            lambda x: any(skill in x for skill in skill_filter) if skill_filter else True
        ))
    ]

    st.write(f"Showing {len(filtered_df)} matching candidates")
    st.dataframe(filtered_df)

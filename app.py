# streamlit_app.py
import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
import re
import time  # For rate limiting
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
import requests
import re
import time
# Configure Gemini
# Load environment variables from .env file
load_dotenv()

# Access the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# ----------------------
# Utility Functions
# ----------------------
@st.cache_data
def load_catalog_data():
    try:
        return pd.read_csv("catalog.csv")
    except Exception as e:
        st.error(f"CSV could not be found or is unreadable: {e}")
        return pd.DataFrame()

def is_valid_url(url):
    pattern = re.compile(
        r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*$'
    )
    return bool(pattern.match(url))

def rate_limited_call(func, *args, **kwargs):
    MAX_RETRIES = 3
    for _ in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except genai.types.StopCandidateException as e:
            st.error(f"API rate limit exceeded. Waiting 10 seconds...")
            time.sleep(10)
    raise Exception("API request failed after multiple retries")

# ----------------------
# Data Processing
# ----------------------
def json_extraction(response_text):
    try:
        match = re.search(r'\[\s*{.*?}\s*]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []

def fetch_description(url):
    try:
        if not is_valid_url(url):
            st.error("Invalid URL format")
            return ""
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        return ' '.join(soup.stripped_strings)
        
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return ""

# ----------------------
# Core Functionality
# ----------------------
def get_assessment_recommendation(query):
    if not query.strip():
        return []
        
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = (
        "You are a helpful assistant. Based on the following job description, recommend up to 10 relevant SHL assessments.\n\n"
        f"{query.strip()}\n\n"
        "Your response MUST be a valid JSON list. Each object should have these keys:\n"
        "- Assessment Name\n- URL\n- Remote Testing Support (Yes/No)\n"
        "- Adaptive/IRT Support (Yes/No)\n- Duration\n- Test Type\n\n"
        "Respond ONLY in valid JSON format like this:\n"
        '[{"Assessment Name": "...", "URL": "...", "Remote Testing Support": "Yes", '
        '"Adaptive/IRT Support": "No", "Duration": "30 mins", "Test Type": "Cognitive"}]'
    )
    
    try:
        response = rate_limited_call(model.generate_content, prompt)
        return response.text.strip() if response.text.strip() else ""
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return ""

# ----------------------
# UI Components
# ----------------------
def main():
    st.title("SHL Assessment Recommender")
    
    # Input Section
    input_type = st.radio("Select input type:", ["Job Description Text", "Job Description URL"])
    job_desc = ""
    
    if input_type == "Job Description Text":
        job_desc = st.text_area("Paste the job description here:")
    elif input_type == "Job Description URL":
        job_url = st.text_input("Enter the job description URL:")
        if job_url:
            if not is_valid_url(job_url):
                st.error("Please enter a valid HTTP/HTTPS URL")
            else:
                job_desc = fetch_description(job_url)
    
    # Recommendation Section
    if st.button("Recommend Assessments") and job_desc.strip():
        with st.spinner("Generating recommendations..."):
            raw_json = get_assessment_recommendation(job_desc)
            
            st.subheader("📦 Raw Gemini Output")
            st.code(raw_json, language="json")
            
            recommendations = json_extraction(raw_json)
            required_keys = {
                "Assessment Name", "URL", "Remote Testing Support",
                "Adaptive/IRT Support", "Duration", "Test Type"
            }
            
            if recommendations:
                valid_recs = [rec for rec in recommendations if required_keys.issubset(rec.keys())]
                if valid_recs:
                    st.subheader("📋 Recommended Assessments")
                    st.table(pd.DataFrame(valid_recs))
                else:
                    st.warning("Invalid recommendation format. Please refine your input.")
            else:
                st.error("❌ Failed to parse recommendations. Try refining your input.")
    
    # Catalog Testing Section
    st.divider()
    st.subheader("🧪 SHL Product Catalog Test")
    
    if st.button("Scrape SHL Catalog"):
        url = "https://www.shl.com/solutions/products/product-catalog/"
        scraped = extract_raw_data(url)
        
        if "error" in scraped:
            st.error(f"Scraping failed: {scraped['error']}")
        else:
            with st.expander("📄 Extracted Text"):
                st.text_area("Raw Text", scraped["text"][:3000], height=300)
            with st.expander("🔗 Links Found"):
                st.write(scraped["links"])

# ----------------------
# Helper Functions
# ----------------------
def extract_raw_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return {
            "text": soup.get_text(separator='\n', strip=True),
            "links": [a['href'] for a in soup.find_all('a', href=True)]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()
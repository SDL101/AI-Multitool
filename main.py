import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import io

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")

def scrape_url(url):
    try:
        page = requests.get(url)
        page.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.
        soup = BeautifulSoup(page.content, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text, None
    except Exception as e:
        return None, str(e)

def extract_text_from_pdf(file):
    pdfReader = PdfReader(io.BytesIO(file.read()))
    all_text = ""
    for page in pdfReader.pages:
        all_text += page.extract_text()
    return all_text

def main():
    
    # Model selection
    model_options = ["bert-base-uncased", "t5-small", "facebook/bart-large-cnn"]
    selected_model = st.sidebar.selectbox("Select Summarizer Model", model_options)

    # Load the chosen model for summarization
    summarizer = pipeline("summarization", model=selected_model)

    # Sidebar for settings
    st.sidebar.title("Settings")
    summary_length = st.sidebar.slider("Summary Length", min_value=25, max_value=250, value=125)
    summary_type = st.sidebar.selectbox("Summary Length Type", ("Words", "Characters"))

    # Main app
    st.title("Full-Stack NLP Text Summarizer Application")
    st.write("Welcome to the Text Summarization App!")

    # Input method selection
    input_method = st.radio("Choose your input method", ("Enter URL", "Paste Text", "Upload PDF"))

    user_input = ""
    if input_method == "Enter URL":
        url_input = st.text_input("Enter URL")
        if url_input:
            scraped_text, error = scrape_url(url_input)
            if error:
                st.error(f"Error occurred while fetching data from URL: {error}")
            else:
                user_input = scraped_text
    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            user_input = extract_text_from_pdf(uploaded_file)
    else:
        user_input = st.text_area("Paste Text Here", "Type or paste text here.")

    if st.button('Summarize'):
        if user_input:
            with st.spinner('Summarizing...'):
                try:
                    summarized_text = summarizer(user_input, max_length=summary_length, min_length=50, do_sample=False)[0]['summary_text']
                    st.text_area("Summarized Text", summarized_text, height=150)
                    st.success("Summarization Complete")
                    st.download_button(label="Download Summary", data=summarized_text, file_name="summary.txt", mime="text/plain")
                except Exception as e:
                    st.error("An error occurred during summarization. Please try again.")
                    st.error("Error details: " + str(e))
        else:
            st.warning("Please enter some text to summarize, provide a URL, or upload a PDF.")

if __name__ == "__main__":
    main()

import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import io
import re
import nltk
from googletrans import Translator

# download the Punkt tokenizer models used for sentence tokenization
nltk.download('punkt')

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")
translator = Translator()
# classifier = pipeline("sentiment-analysis")

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
        text = page.extract_text() + ' ' if page.extract_text() else ''
        # Replace newlines with spaces and add space at the end to separate pages
        all_text += text.replace('\n', ' ')
    # Reduce multiple spaces to a single space
    all_text = re.sub(' +', ' ', all_text)
    return all_text

def clean_and_capitalize(text):
    # Remove any space immediately before a period, exclamation point, or question mark
    text = re.sub(r'\s+([.!?])', r'\1', text)
    
    # Ensure that each sentence ending is properly spaced and the following sentence starts capitalized
    text = re.sub(r'([.!?])\s*([a-zA-Z])', lambda match: match.group(1) + ' ' + match.group(2).upper(), text)
    
    # Ensure the very first letter of the text is capitalized
    if text:
        text = text[0].upper() + text[1:]

    return text

def add_space_after_periods(text):
    # This will add a space after periods where it is not followed by a space
    return re.sub(r'\.(?![\s|$])', '. ', text)

def ensure_full_sentences(summary):
    sentences = nltk.tokenize.sent_tokenize(summary)
    if sentences:
        return ' '.join(sentences[:-1]) + ' ' + sentences[-1] if len(sentences) > 1 else sentences[0]
    return summary

def main():
    if 'cleaned_text' not in st.session_state:
        st.session_state['cleaned_text'] = ""
    if 'translated_text' not in st.session_state:  # Initialize the translated text in session state
        st.session_state['translated_text'] = ""
    
    # Model selection
    # model_options = ["bert-base-uncased", "t5-small", "facebook/bart-large-cnn"]
    model_options = ["Default Model", "t5-small"]
    selected_model = st.sidebar.selectbox("Select Summarizer Model", model_options)
    # Conditionally load the chosen model for summarization
    if selected_model == "Default Model":
        # Load the default model
        summarizer = pipeline("summarization")
    else:
        # Load the specified model
        summarizer = pipeline("summarization", model=selected_model)

    # Sidebar for settings
    st.sidebar.title("Customization Options")
    summary_length_option = st.sidebar.selectbox("Choose Summary Length", options=["Short", "Medium", "Long"])
    # st.write(f"Current summary length: {summary_length}")
    # summary_type = st.sidebar.selectbox("Summary Length Type", ("Words", "Characters"))

    # Main app
    st.title("Text Summarizer with Summary Translation")
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
            extracted_text = extract_text_from_pdf(uploaded_file)
            # Print out the extracted PDF contents
            st.text_area("Extracted Text", extracted_text, height=300)  # Adjust height as needed
            user_input = extracted_text
    else:
        user_input = st.text_area("Paste Text Here", "Type or paste text here.")

    if st.button('Summarize'):
        if user_input:
            with st.spinner('Summarizing...'):
                try:
                    # Dynamically adjust summary length based on input text length
                    input_length = len(user_input.split())
                    if summary_length_option == "Short":
                        summary_length = max(10, int(input_length / 25))
                    elif summary_length_option == "Medium":
                        summary_length = max(25, int(input_length / 10))
                    else:  # Long
                        summary_length = max(50, int(input_length / 3))
                    
                    st.write(f"Input Length: {input_length} words")
                    st.write(f"Summary Length Option: {summary_length_option}")
                    st.write(f"Calculated Summary Length: {summary_length} tokens")
                    st.write(f"Max Length: {max(0,summary_length+25)}, Min Length: {max(0, summary_length-25)}")

                    # Generate summary
                    summarized_text = summarizer(user_input, max_length=max(0,summary_length+25), min_length=max(0, summary_length-25), do_sample=False)[0]['summary_text']
                    sum_len = len(summarized_text.split())
                    st.write(f"Length of summary in words: {sum_len}")
                    # Ensure summary ends with a full sentence
                    st.write("Summary PRE-call to ensure_full_sentences: ", summarized_text)
                    summarized_text = ensure_full_sentences(summarized_text)
                    st.write("Summary POST-call to ensure_full_sentences: ", summarized_text)
                    st.session_state['cleaned_text'] = clean_and_capitalize(summarized_text)
                    # Display summarized text from session state
                    st.text_area("Summarized Text", st.session_state['cleaned_text'], height=150)
                    # Display download button for the original English summary
                    st.download_button(
                        label="Download English Summary",
                        data=st.session_state['cleaned_text'],
                        file_name="english_summary.txt",
                        mime="text/plain"
                    )
                    st.success("Summarization Complete")
                except Exception as e:
                    st.error("An error occurred during summarization. Please try again.")
                    st.error("Error details: " + str(e))
        else:
            st.warning("Please enter some text to summarize, provide a URL, or upload a PDF.")
    else:
        # If the page is not being summarized, display existing summarized text (if any)
        if st.session_state['cleaned_text']:
            st.text_area("Summarized Text", st.session_state['cleaned_text'], height=150)
            # Display download button for the existing English summary
            st.download_button(
                label="Download English Summary As Text File",
                data=st.session_state['cleaned_text'],
                file_name="english_summary.txt",
                mime="text/plain"
            )

    # Translation Section
    if st.session_state['cleaned_text']:
        st.write("Translate Summary To Another Language")
        languages = {"Spanish": "es", "Mandarin": "zh-CN", "Japanese": "ja", "French": "fr"}
        target_language = st.selectbox("Choose a language", list(languages.keys()))
        if st.button("Translate"):
            with st.spinner('Translating...'):
                translated_result = translator.translate(st.session_state['cleaned_text'], src='en', dest=languages[target_language])
                # Apply add_space_after_periods only for languages that use Latin script
                if languages[target_language] in ["es", "fr"]:  # Add more language codes if needed
                    translated_text = add_space_after_periods(translated_result.text)
                else:
                    translated_text = translated_result.text
                st.session_state['translated_text'] = translated_text
                # Display translated text and download button for the translated summary
                st.text_area(f"Translated Summary ({target_language})", st.session_state['translated_text'], height=150)
                st.download_button(
                    label=f"Download Translated {target_language} Summary As Text File",
                    data=st.session_state['translated_text'],
                    file_name=f"translated_summary_{target_language}.txt",
                    mime="text/plain"
                )
    elif 'translated_text' in st.session_state and st.session_state['translated_text']:
        # If a translation has already been made, keep showing the translated text and its download button
        st.text_area(f"Translated Summary ({target_language})", st.session_state['translated_text'], height=150)
        st.download_button(
            label=f"Download Translated {target_language} Summary As Text File",
            data=st.session_state['translated_text'],
            file_name=f"translated_summary_{target_language}.txt",
            mime="text/plain"
        )
    else:
        st.session_state['translated_text'] = ""

if __name__ == "__main__":
    main()

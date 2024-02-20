import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import io
import re
import nltk
from googletrans import Translator

# Set the page to wide mode by default (can be updated in settings)
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Remove top margin and padding from the main container */
    .main .block-container {
        padding-top: 0;
        margin-top: 0;
    }
    /* Adjust the top margin of the title specifically, if needed */
    .title {
        margin-top: 5px; /* Reduce top margin to bring the title closer to the top */
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS to inject into Streamlit's HTML
st.markdown("""
    <style>
    /* Target the standard button's hover state */
    .stButton>button:hover {
        border-color: #4a69bd; /* Border color on hover */
        background-color: #4a69bd; /* Button color on hover */
        color: #ffffff; /* Text color on hover */
    }
    /* Target the download button's hover state specifically, if needed */
    .stDownloadButton>button:hover {
        border-color: #4a69bd; /* Border color on hover */
        background-color: #4a69bd; /* Button color on hover */
        color: #ffffff; /* Text color on hover */
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .title {
        font-size: 40px; /* Larger font size for the title */
        font-weight: bold;
        color: #333; /* Adjust the color as needed */
        margin-bottom: 0px; /* Adds some space below the title */
    }
    .headline {
        font-size: 20px; /* Smaller font size for the headline */
        color: #333; /* Adjust the color as needed */
        margin-bottom: 20px; /* Adds some space below the title */
    }
    </style>
    <div class="title">Information Multitool</div>
    <div class="headline">Engineered with ❤️ by Scott Lindsay</div>
    """, unsafe_allow_html=True)


# download the Punkt tokenizer models used for sentence tokenization
nltk.download('punkt')

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")
translator = Translator()
# classifier = pipeline("sentiment-analysis")

st.write("""APP TO DO:\n
        1. add more languages\n
        2. when you change lengths reload/wipe the summary\n 
        3. """)

def scrape_url(url):
    try:
        headers = {
            'Accept-Encoding': 'identity',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text, None
    except requests.exceptions.HTTPError as e:
        return None, f'HTTP Error: {e.response.status_code}'
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
    if not sentences:  # If no sentences, return empty summary
        return summary
    
    # Check if the last sentence ends with a proper punctuation
    if sentences[-1].endswith(('.', '!', '?')):
        return ' '.join(sentences)
    else:
        return ' '.join(sentences[:-1])  # Exclude the last incomplete sentence

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

    # # Main app
    # st.title("Text Summarizer with Summary Translation")
    # st.write("Welcome to the Text Summarization App!")

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
                # Print out the scraped text for inspection
                st.text_area("Scraped Text", user_input, height=300)
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
            if len(user_input) > 1024:  # Example threshold, adjust as needed
                user_input = user_input[:1024]  # Truncate to first 1024 characters
            with st.spinner('Summarizing...'):
                try:
                    # Dynamically adjust summary length based on input text length
                    input_length = len(user_input.split())
                    if summary_length_option == "Short":
                        summary_length = max(25, int(input_length / 20))
                    elif summary_length_option == "Medium":
                        summary_length = max(50, int(input_length / 10))
                    else:  # Long
                        summary_length = max(100, int(input_length / 3))
                    
                    stats_message = (f"**Input Length:** {input_length} words,\n"
                                     f"**Summary Length Option:** {summary_length_option},\n"
                                     f"**Calculated Summary Length:** {summary_length} tokens,\n"
                                     f"**Max Length:** {summary_length+50} tokens,\n"
                                     f"**Min Length:** {summary_length-25} tokens.")
                    st.markdown("##### Summary Request Details: ")
                    st.markdown(stats_message)

                    # Generate summary
                    summarized_text = summarizer(user_input, max_length=max(0,summary_length+50), min_length=max(0, summary_length-25), do_sample=False)[0]['summary_text']
                    sum_len = len(summarized_text.split())
                    st.write(f"**Length of summary:** {sum_len} words.")
                    # Ensure summary ends with a full sentence
                    # st.write("Summary PRE-call to ensure_full_sentences: ", summarized_text)
                    summarized_text = ensure_full_sentences(summarized_text)
                    # st.write("Summary POST-call to ensure_full_sentences: ", summarized_text)
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
                    st.balloons() #add some user feedback
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

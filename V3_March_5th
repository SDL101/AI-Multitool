# Importing necessary libraries for the web application
import streamlit as st  # Streamlit library for creating web applications
from transformers import pipeline  # Hugging Face's transformers library for NLP tasks
import requests  # Requests library to make HTTP requests
from bs4 import BeautifulSoup  # BeautifulSoup library for parsing HTML and XML documents
from PyPDF2 import PdfReader  # PyPDF2 library for reading PDF files
import io  # io library for handling various types of I/O
import re  # re library for regular expression operations
import nltk  # nltk library for natural language processing tasks
from googletrans import Translator  # googletrans library for language translation

# Configuring the Streamlit web page
# Set the page to wide mode by default (can be updated in settings)
st.set_page_config(layout="wide")


# Injecting custom CSS to remove top margin and padding from the main container for a cleaner look
st.markdown("""
    <style>
    /* Remove top margin and padding from the main container */
    .main .block-container {
        padding-top: 10;
        margin-top: 0;
    }
    /* Adjust the top margin of the title specifically, if needed */
    .title {
        margin-top: 5px; /* Reduce top margin to bring the title closer to the top */
    }
    </style>
    """, unsafe_allow_html=True)  # unsafe_allow_html allows the use of HTML within the markdown function

# Custom CSS to inject into Streamlit's HTML for persistent button color across the application
st.markdown("""
    <style>
    /* Target the standard Streamlit button */
    .stButton>button {
        border-color: #4a69bd; /* Custom border color */
        background-color: #4a69bd; /* Custom button color */
        color: #ffffff; /* Custom text color */
    }
    /* Target the download button specifically, if needed, to maintain consistency */
    .stDownloadButton>button {
        border-color: #4a69bd; /* Custom border color */
        background-color: #4a69bd; /* Custom button color */
        color: #ffffff; /* Custom text color */
    }
    </style>
    """, unsafe_allow_html=True)  # This again allows the use of HTML for styling purposes

# Adding custom styles and content for the application's title and subtitle
st.image('logo.jpeg', caption="Engineered with ❤️ by Scott Lindsay", width=250)
# st.markdown("""
#     <style>
#     .title {
#         font-size: 40px; /* Larger font size for the title to make it prominent */
#         font-weight: bold; /* Bold font weight for emphasis */
#         color: #4a69bd; /* Custom color for the title */
#         margin-bottom: 0px; /* Adjust bottom margin to control space below the title */
#     }
#     .headline {
#         font-size: 10px; /* Smaller font size for subtitles or headlines for distinction */
#         color: #333; /* Custom color for the headline, usually more subdued than the title */
#         margin-bottom: 20px; /* Bottom margin to add space below the headline */
#     }
#     </style>
#     """, unsafe_allow_html=True)  # Allows HTML to enable detailed styling and layout control

# Section: Natural Language Processing and Web Scraping Utilities

# Download the Punkt tokenizer models used for sentence tokenization
nltk.download('punkt')

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")
classifier = pipeline("sentiment-analysis")

# Initialize the Google Translate API client for translation tasks
translator = Translator()

# Define a function to scrape text content from a given URL
def scrape_url(url):
    try:
        # Set custom headers to simulate a request from a web browser
        headers = {
            'Accept-Encoding': 'identity',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        # Make a GET request to the URL
        response = requests.get(url, headers=headers)
        # Raise an HTTPError if the response was an error
        response.raise_for_status()
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract and concatenate text from all paragraph tags
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        # Return the scraped text and None for error
        return text, None
    except requests.exceptions.HTTPError as e:
        # Return None and HTTP error message if an HTTPError is caught
        return None, f'HTTP Error: {e.response.status_code}'
    except Exception as e:
        # Return None and the error message for any other exceptions
        return None, str(e)

# Define a function to extract text from a PDF file
def extract_text_from_pdf(file):
    # Initialize a PdfReader object with the PDF file
    pdfReader = PdfReader(io.BytesIO(file.read()))
    all_text = ""
    # Iterate through each page in the PDF
    for page in pdfReader.pages:
        # Extract text from the page, append a space if text is found
        text = page.extract_text() + ' ' if page.extract_text() else ''
        # Replace newlines with spaces and add space at the end to separate pages
        all_text += text.replace('\n', ' ')
    # Reduce multiple spaces to a single space
    all_text = re.sub(' +', ' ', all_text)
    # Return the concatenated text from all pages
    return all_text

def clean_and_capitalize(text):
    """
    Cleans and capitalizes the text by removing unnecessary spaces before punctuation marks,
    ensuring proper sentence spacing and capitalization.
    """
    # Remove any space immediately before a period, exclamation point, or question mark
    text = re.sub(r'\s+([.!?])', r'\1', text)
    
    # Ensure that each sentence ending is properly spaced and the following sentence starts capitalized
    text = re.sub(r'([.!?])\s*([a-zA-Z])', lambda match: match.group(1) + ' ' + match.group(2).upper(), text)
    
    # Ensure the very first letter of the text is capitalized
    if text:
        text = text[0].upper() + text[1:]

    return text

def add_space_after_periods(text):
    """
    Adds a space after periods where it is not followed by a space,
    to ensure proper sentence separation.
    """
    # This will add a space after periods where it is not followed by a space
    return re.sub(r'\.(?![\s|$])', '. ', text)

def ensure_full_sentences(summary):
    """
    Processes the summary to ensure it consists of full sentences.
    Incomplete sentences at the end of the summary are removed.
    """
    # Tokenize the summary into sentences
    sentences = nltk.tokenize.sent_tokenize(summary)
    
    # If no sentences, return the original summary
    if not sentences:
        return summary
    
    # Check if the last sentence ends with a proper punctuation mark (period, exclamation point, or question mark)
    if sentences[-1].endswith(('.', '!', '?')):
        # If it does, return the joined full sentences
        return ' '.join(sentences)
    else:
        # If the last sentence is incomplete, exclude it and return the rest
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
                    summarized_text = ensure_full_sentences(summarized_text)
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
            # Sentiment analysis
            if st.session_state['cleaned_text']:
                sentiment_result = classifier(st.session_state['cleaned_text'])[0]
                sentiment_label = sentiment_result['label']
                sentiment_score = sentiment_result['score']
            
            # Display sentiment analysis results
            st.write(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
                
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
    # Check if there's cleaned text available in the session state to translate
    if st.session_state['cleaned_text']:
        # Display a header or prompt in the app for the translation feature
        st.write("Translate Summary To Another Language")

        # Define a dictionary mapping language names to their respective language codes
        # These codes are used by the translation service to identify target languages
        languages = {
            "Spanish": "es",  # Spanish language code
            "Mandarin": "zh-CN",  # Mandarin Chinese, specifying Simplified Chinese script
            "English": "en",  # English language code, included for completeness or re-translation
            "Hindi": "hi",  # Hindi language code
            "Bengali": "bn",  # Bengali language code
            "Portuguese": "pt",  # Portuguese language code
            "Russian": "ru",  # Russian language code
            "Japanese": "ja",  # Japanese language code
            "Western Punjabi": "pa",  # Punjabi, specifically Western Punjabi spoken in Pakistan
            "Marathi": "mr",  # Marathi language code
            "French": "fr"  # French language code
        }

            # Allow the user to select a target language from a dropdown list
        target_language = st.selectbox("Choose a language", list(languages.keys()))

        # Create a button that, when clicked, will initiate the translation process
        if st.button("Translate"):
            # Show a spinner animation during the translation process to indicate that work is being done
            with st.spinner('Translating...'):
                # Call the translate method from a translator object, specifying source and destination languages
                # 'src' is the source language code, 'dest' is the target language code from the selected language
                translated_result = translator.translate(st.session_state['cleaned_text'], src='en', dest=languages[target_language])
                
                # Some languages use Latin script and may need additional formatting, such as adding spaces after periods
                latin_script_languages = ["es", "pt", "fr", "en"]  # Define which languages use Latin script
                
                # Check if the target language uses Latin script and apply formatting if necessary
                if languages[target_language] in latin_script_languages:
                    translated_text = add_space_after_periods(translated_result.text)
                else:
                    translated_text = translated_result.text
                
                # Store the translated text in the session state for display and potential download
                st.session_state['translated_text'] = translated_text
                
                # Display the translated text in a text area widget, allowing the user to review it
                st.text_area(f"Translated Summary ({target_language})", st.session_state['translated_text'], height=150)
                
                # Provide a download button for the user to download the translated text as a .txt file
                st.download_button(
                    label=f"Download Translated {target_language} Summary As Text File",
                    data=st.session_state['translated_text'],
                    file_name=f"translated_summary_{target_language}.txt",
                    mime="text/plain"
                )

        # Check if there is already translated text available in the session state
        elif 'translated_text' in st.session_state and st.session_state['translated_text']:
            # Display the previously translated text and the download button again
            # This ensures that the translated text remains visible even after the page is refreshed or updated
            st.text_area(f"Translated Summary ({target_language})", st.session_state['translated_text'], height=150)
            st.download_button(
                label=f"Download Translated {target_language} Summary As Text File",
                data=st.session_state['translated_text'],
                file_name=f"translated_summary_{target_language}.txt",
                mime="text/plain"
            )
        else:
            # Initialize the 'translated_text' in session state as an empty string if not already set
            # This is a fallback to ensure the variable is initialized before its first use
            st.session_state['translated_text'] = ""

if __name__ == "__main__":
    main()

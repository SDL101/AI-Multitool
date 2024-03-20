import streamlit as st  # Streamlit library for creating web applications
from transformers import pipeline  # Hugging Face's transformers library for NLP tasks
import requests  # Requests library to make HTTP requests
from bs4 import BeautifulSoup  # BeautifulSoup library for parsing HTML and XML documents
from PyPDF2 import PdfReader  # PyPDF2 library for reading PDF files
import io  # io library for handling various types of I/O
import re  # re library for regular expression operations
import nltk  # nltk library for natural language processing tasks
from googletrans import Translator  # googletrans library for language translation

# Set the page to wide mode by default (can be updated in settings)
st.set_page_config(layout="wide")

# Injecting custom CSS to adjust app appearance and element styles
st.markdown("""
    <style>
    /* Adjust top margin and padding of the main container for a cleaner look */
    .main .block-container {
        padding-top: 20px !important; /* Reduce top padding to remove space above the logo */
        margin-top: 20px !important; /* Reduce top margin to remove space above the logo */
    }
    /* Adjust the top margin of the title specifically, if needed */
    .title {
        margin-top: 20px !important; /* Reduce top margin to bring the title closer to the top */
    }
    /* Customize the standard Streamlit button appearance */
    .stButton>button {
        border-color: #4a69bd; /* Custom border color */
        background-color: #4a69bd; /* Custom button color */
        color: #ffffff; /* Custom text color */
    }
    /* Ensure consistent styling for the download button */
    .stDownloadButton>button {
        border-color: #4a69bd; /* Maintain custom border color */
        background-color: #4a69bd; /* Maintain custom button color */
        color: #ffffff; /* Maintain custom text color */
    }
    </style>
    """, unsafe_allow_html=True)

# Adding custom styles and content for the application's title and subtitle
st.image('logo.jpeg', width=250)

# Download the Punkt tokenizer models used for sentence tokenization
nltk.download('punkt')

# Initialize the Hugging Face summarization and classification piplines
summarizer = pipeline("summarization")
classifier = pipeline("sentiment-analysis")

# Initialize the Google Translate API client for translation tasks
translator = Translator()

# Define a function to scrape text content from a given URL
def scrape_url(url):
    """
    Scrapes and returns the textual content from the specified URL by extracting text from all paragraph elements.

    The function makes a GET request to the given URL with custom headers that simulate a request from a web browser, 
    enhancing compatibility and access. It then parses the HTML content of the page to extract and concatenate text 
    content from all paragraph (<p>) tags found within the HTML structure. In case of a successful text extraction, 
    the concatenated text is returned along with `None` for the error. If an HTTP error occurs during the request, 
    the function captures the HTTP error and returns `None` for the text and the error message. For any other 
    exceptions that might occur during the process, it also returns `None` for the text and a generic error message.

    Parameters:
    - url (str): The URL of the webpage from which to scrape text.

    Returns:
    - tuple: A tuple containing two elements. The first element is the scraped text (str) or `None` if an error occurred.
             The second element is `None` for a successful scrape or an error message (str) if an error occurred.

    Note:
    - This function requires the `requests` library for making HTTP requests and the `BeautifulSoup` class from `bs4`
      for parsing HTML content. Ensure these dependencies are installed and imported before using this function.
    - Custom headers, including a User-Agent string, are used to mimic a web browser request, which can help in
      accessing webpages that may restrict access to scripts or bots.
    - It handles specific exceptions such as HTTP errors gracefully by providing error messages, aiding in debugging 
      and error handling.
    """
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

def extract_text_from_pdf(file):
    """
    Extracts and concatenates text from all pages of an uploaded PDF file.

    This function initializes a PdfReader object with the provided PDF file, iterates through each page of the PDF,
    and extracts the text. The extracted text from each page is concatenated into a single string. During concatenation,
    newline characters are replaced with spaces to maintain readability and ensure that text from different lines and pages
    flows smoothly without unintended concatenation of words. Additionally, multiple spaces are reduced to a single space
    to clean up any irregular spacing in the text extraction process. This function returns the cleaned, concatenated text string.

    Parameters:
    - file (file-like object): The PDF file from which text is to be extracted. The file should be opened in binary mode.

    Returns:
    - str: The concatenated text extracted from all pages of the PDF.

    Note:
    - The function requires the PdfReader class from a PDF processing library (e.g., PyPDF2) and the re module for regular expressions.
    - It assumes that the `file` parameter is a file-like object that supports the `.read()` method, typically passed as an uploaded file in web frameworks.
    """
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
    Cleans the provided text by removing unnecessary spaces before punctuation marks and capitalizes the first character of each sentence.

    The function performs three main operations to clean and capitalize the text:
    1. It removes any spaces that appear immediately before punctuation marks such as periods, exclamation points, or question marks,
       ensuring that punctuation is correctly positioned without preceding spaces.
    2. It ensures that after each sentence-ending punctuation mark (period, exclamation point, or question mark), the following
       sentence starts with a capital letter. This is done by capitalizing the first character following the punctuation if it is
       a letter, and ensuring proper spacing between sentences.
    3. The very first letter of the entire text is capitalized, if the text is not empty, to ensure that the text starts with a capital letter.

    Parameters:
    - text (str): The text to be cleaned and capitalized.

    Returns:
    - str: The cleaned and capitalized text.

    Note:
    - The function uses regular expressions (the re module) for text manipulation, which must be imported before using this function.
    - This method is particularly useful for formatting text to be more readable or to adhere to stylistic conventions, especially
      after text extraction processes where spacing and capitalization may be inconsistent.
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
    Modifies the provided text to ensure there is a space following each period, except in cases where a space or the end of the text already follows the period.

    This function addresses a common formatting issue where sentences are not properly separated due to missing spaces after periods. By applying a regular expression,
    the function identifies periods that are not followed by a space or are at the end of the text. It then inserts a space immediately after each identified period 
    to ensure correct sentence separation. This process helps improve the readability of the text by maintaining standard spacing conventions between sentences.

    Parameters:
    - text (str): The text to be modified for improved sentence separation.

    Returns:
    - str: The modified text with spaces added after periods where necessary.

    Note:
    - The function relies on the `re` module for regular expression operations, which should be imported before using this function.
    - This method is useful for preprocessing text for readability or further natural language processing tasks where sentence boundary detection is important.
    """
    # This will add a space after periods where it is not followed by a space
    return re.sub(r'\.(?![\s|$])', '. ', text)

def ensure_full_sentences(summary):
    """
    Enhances the coherence of a summary by ensuring that it is composed only of full sentences. Any incomplete sentence found at the end of the summary is removed.

    This function first tokenizes the provided summary into individual sentences using the Natural Language Toolkit (NLTK). It then examines the last sentence to 
    determine whether it ends with a proper punctuation mark (period, exclamation point, or question mark). If the last sentence is deemed complete, the original 
    tokenized sentences are rejoined and returned as the final summary. However, if the last sentence does not end with a proper punctuation mark, indicating it 
    is incomplete, it is removed from the list of sentences. The remaining sentences are then rejoined and returned, ensuring that the summary does not end on 
    an incomplete thought.

    Parameters:
    - summary (str): The text summary to be processed for completeness.

    Returns:
    - str: The processed summary, guaranteed to end with a full sentence or be returned unchanged if it consists solely of full sentences.

    Note:
    - This function requires NLTK for sentence tokenization (`nltk.tokenize.sent_tokenize`). Ensure NLTK is installed and imported.
    - Ideal for processing textual summaries or content where ending on partial sentences could detract from readability or clarity.
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
    """
    Serves as the main entry point for a web-based text summarization and translation application. 

    This function orchestrates a series of steps to enable users to input text (via URL, pasting directly, or uploading a PDF), 
    select a summarization model, customize summary length, and optionally translate the summarized text into a different language. 
    It initializes session states for storing processed texts, allows model selection for text summarization, provides input options 
    for users, and implements text summarization with dynamic length based on user preference. Post summarization, it offers sentiment 
    analysis of the summarized content and a translation feature for the summary into multiple languages. Throughout this process, 
    user feedback is provided through the application interface, including error messages, success confirmations, and an interactive 
    spinner during processing stages.

    The function leverages the Streamlit library to create and manage the web interface, including session state for storing text, 
    sidebar options for customization, and buttons for triggering summarization and translation. It relies on external libraries 
    such as `requests` and `BeautifulSoup` for web scraping, `transformers` for text summarization, and a sentiment analysis model 
    for evaluating the tone of the summarized text. Translation is accomplished using a third-party API or library, adjusting for 
    language-specific formatting where necessary.

    Error handling is integrated to manage exceptions during web scraping, PDF text extraction, summarization, and translation, 
    ensuring the application remains robust and user-friendly. The function also provides options for downloading the summarized 
    and translated text, enhancing the application's utility.

    Note:
    - The detailed implementation of text extraction from PDFs, web scraping, summarization, and translation are abstracted 
      away in this description and rely on respective helper functions and external services.
    - Ensure all necessary libraries and external services are available and properly configured, including Streamlit for 
      the web interface, `transformers` for summarization, and any required libraries for translation and sentiment analysis.
    - This function is designed to be used within a Streamlit application, and its execution is triggered by running the 
      Streamlit app.
    """
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
    input_method = st.radio("Choose your input method", ("Enter URL", "Enter/Paste Text", "Upload PDF"))

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
        user_input = st.text_area("Enter or Paste Text Here", "Type or paste text here.")
    # elif input_method == "Paste Text":
    #     user_input_raw = st.text_area("Paste Text Here", "Type or paste text here.")
    #     user_input = user_input_raw.strip()  # Trim whitespace from the input

    if st.button('Summarize'):
        if user_input:
            # st.write(len(user_input))
            if len(user_input) > 1024:  # Example threshold, adjust as needed
                user_input = user_input[:1024]  # Truncate to first 1024 characters
            elif len(user_input) < 50:
                st.error("Error, invalid input. Please ensure your input is not empty, in English, and at least a few sentences in length. You can enter text directly, provide a URL, or upload a PDF.")
                return
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

                    # Generate summary
                    summarized_text = summarizer(user_input, max_length=max(0,summary_length+50), min_length=max(0, summary_length-25), do_sample=False)[0]['summary_text']
                    sum_len = len(summarized_text.split())
                    with st.expander("Show Technical Details"):
                        st.markdown("##### Summary Request Details: ")
                        stats_message = (f"**Input Length:** {input_length} words,\n"
                                        f"**Summary Length Option:** {summary_length_option},\n"
                                        f"**Calculated Summary Length:** {summary_length} tokens,\n"
                                        f"**Max Length:** {summary_length+50} tokens,\n"
                                        f"**Min Length:** {summary_length-25} tokens.\n"
                                        f"**Length of summary:** {sum_len} words.")
                        st.markdown(stats_message)
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
                    st.error("An error occurred during summarization. Please refresh the page and try again.")
                    st.error("Error details: " + str(e))
            # Sentiment analysis
            if st.session_state['cleaned_text']:
                sentiment_result = classifier(st.session_state['cleaned_text'])[0]
                sentiment_label = sentiment_result['label']
                sentiment_score = sentiment_result['score']
            
            # Display sentiment analysis results
            st.write(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
                
        else:
            st.error("Error, invalid input. Please ensure your input is not empty, in English, and at least a few sentences in length. You can enter text directly, provide a URL, or upload a PDF.")
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
    if st.session_state['cleaned_text']:   # Check if cleaned text in session state to translate
        st.write("Translate Summary To Another Language") # Display header or prompt in app for translation feature

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
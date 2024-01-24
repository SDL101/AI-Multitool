import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")

def main():
    st.title("Full-Stack NLP Text Summarizer Application by Scotty Lindsay for CS469")
    st.write("Welcome to the rough draft UI outline for my Text Summarization App! Once implemented, this tool will help you summarize your text.")

    # Text input
    user_input = st.text_area("Enter Text Here", "Type or paste the text you want to summarize.")

    # Summarization options
    st.write("Customize your summarization:")
    summary_length = st.slider("Select Summary Length", min_value=10, max_value=500, value=100)
    summary_type = st.selectbox("Summary Length Type", ("Words", "Characters"))

    # Summarize button
    if st.button('Summarize'):
        # Perform summarization
        summarized_text = summarizer(user_input, max_length=summary_length, min_length=10, do_sample=False)[0]['summary_text']
        # Display summarized text
        st.text_area("Summarized Text", summarized_text, height=150)

if __name__ == "__main__":
    main()

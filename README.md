# Visual Story teller

AI Idea Generator

This project consists of two main Python scripts: app7.py and app8.py.
app7.py

This script contains the core functions for the AI Idea Generator. It includes functions to extract text from YouTube videos and PDF files, process the extracted text, and generate ideas based on a given query.

Here are the main functions defined in app7.py:

- extract_raw_text_from_video(url): Extracts the raw text from a YouTube video given its URL.
- extract_pdf_text(pdf_path): Extracts the text from a PDF file given its path.
- process_text_and_create_faiss_db(video_text, pdf_text): Processes the extracted text and creates a FAISS database for similarity search.
- generate_ideas_with_formatted_prompt(docs): Generates ideas based on the query results.
- clear_downloads(): Clears the downloaded files.
app8.py

This script is a Streamlit app that provides a user interface for the AI Idea Generator. Users can upload a PDF file, enter a YouTube URL, and enter a query. When the 'Generate Idea' button is clicked, the app uses the functions from app7.py to extract text from the provided PDF file and YouTube video, process the extracted text, and generate an idea based on the entered query.

Here's how to run the Streamlit app:

1. Install Streamlit if you haven't already: pip install streamlit
2. Run the app: streamlit run app8.py
3. Open the provided local URL in your web browser.

Please note that you need to have the necessary API keys and other environment variables set up for the functions in app7.py to work correctly.
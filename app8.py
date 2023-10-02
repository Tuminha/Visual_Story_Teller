import streamlit as st
from app7 import extract_raw_text_from_video, extract_pdf_text, process_text_and_create_faiss_db, generate_ideas_with_formatted_prompt, clear_downloads

# Title
st.title('AI Idea Generator 🤖💡')

# Instructional Text
st.markdown("""
    Welcome to the AI Idea Generator! This tool uses advanced AI to help you generate creative ideas based on text from PDFs or YouTube videos. 
    Just upload a PDF, enter a YouTube URL, or do both! Then, enter a query to focus the AI on a particular idea or theme.
    """)

# Input Fields with Labels and Instructions
st.markdown("### Step 1: Input Sources 📤")
pdf_path = st.file_uploader("Drop here a book, articles, or something that inspired you 📚", type=['pdf'])
url = st.text_input('Or paste here a link of a YouTube video that inspired you 🎥')
st.markdown("### Step 2: Guide the AI 🎯")
query = st.text_input('Enter a query to extract the main idea or quotes. For example: "What is the main message of this content?"')

# Generate Idea Button
if st.button('Generate Idea 🚀'):
    clear_downloads()  # Clear previous downloads

    # Process and Fetch Data
    if url:
        video_text = extract_raw_text_from_video(url)
    else:
        video_text = ""

    if pdf_path:
        pdf_text = extract_pdf_text(pdf_path)
    else:
        pdf_text = ""

    vectordb = process_text_and_create_faiss_db(video_text, pdf_text)
    docs = vectordb.similarity_search(query, max_tokens=1500)
    idea = generate_ideas_with_formatted_prompt(docs)

    # Display Results
    st.markdown("### Generated Idea 💡")
    st.write(idea)
    st.write("Based on insights shared by Jun Han Chin 👏")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name or Company]")

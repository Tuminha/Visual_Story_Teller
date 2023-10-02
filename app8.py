import streamlit as st
from app7 import extract_raw_text_from_video, extract_pdf_text, process_text_and_create_faiss_db, generate_ideas_with_formatted_prompt, clear_downloads

# Title
st.title('AI Idea Generator')

# Input fields
pdf_path = st.file_uploader("Choose a PDF file", type=['pdf'])
url = st.text_input('Enter a YouTube URL')
query = st.text_input('Enter a query')

# Button to generate idea
if st.button('Generate Idea'):
    clear_downloads()  # Clear downloads when the button is clicked

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

    # Display the idea
    st.write(idea)
    # Together with the idea output, display a message that says "Based on insights shared by Jun Han Chin"
    st.write("Based on insights shared by Jun Han Chin")
    
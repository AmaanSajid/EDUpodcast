import streamlit as st
from google.cloud import aiplatform
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import base64
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import numpy as np
import faiss
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
aiplatform.init(project="focus-ensign-437302-v1",location="us-central1")

def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def display_pdf(pdf_file):
    # Reset file pointer to beginning
    pdf_file.seek(0)
    
    # Read PDF file
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    # Embed PDF viewer
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="800px"
            type="application/pdf">
        </iframe>
    """
    return pdf_display

def generate_summary(text):
    model = GenerativeModel("gemini-1.5-pro-002")
    prompt = f"""
    Please provide a comprehensive summary of approximately 500 words for the following text. 
    Focus on the main points and key insights:

    {text}
    """
    
    response = model.generate_content(prompt)
    return response.text

def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(chunks):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = []
    chunk_ids = []
    
    for i, chunk in enumerate(chunks):
        embedding = model.get_embeddings([chunk])[0]
        embeddings.append(embedding.values)
        chunk_ids.append(str(i))
    
    return np.array(embeddings).astype('float32'), chunk_ids

def create_and_save_faiss_index(embeddings, ids, index_path, ids_path):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    with open(ids_path, 'w') as f:
        json.dump(ids, f)
    
    return index, ids

def save_chunks(chunks, chunks_path):
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f)

def load_faiss_index(index_path, ids_path):
    index = faiss.read_index(index_path)
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    return index, ids

def search_similar_chunks(query, index, ids, chunks, k=5):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    query_embedding = np.array([model.get_embeddings([query])[0].values]).astype('float32')
    
    D, I = index.search(query_embedding, k)
    
    relevant_chunks = [chunks[int(ids[i])] for i in I[0]]
    return " ".join(relevant_chunks)

def generate_answer(question, context):
    model = GenerativeModel("gemini-1.5-pro-002")
    prompt = f"""
    Based on the following context from a PDF document:
    
    {context}
    
    Please answer this question concisely and accurately: {question}
    Use only information from the provided context.
    """
    
    response = model.generate_content(prompt)
    return response.text

def process_pdf(uploaded_file):
    # Extract text
    text = extract_text_from_pdf(uploaded_file)
    
    # Generate summary
    summary = generate_summary(text)
    
    # Split into chunks
    chunks = split_text(text)
    
    # Generate embeddings
    embeddings, chunk_ids = generate_embeddings(chunks)
    
    # Save everything
    index_path = 'faiss_index.bin'
    ids_path = 'chunk_ids.json'
    chunks_path = 'chunks.json'
    
    index, _ = create_and_save_faiss_index(embeddings, chunk_ids, index_path, ids_path)
    save_chunks(chunks, chunks_path)
    
    return index_path, ids_path, chunks_path, summary

def main():
    st.title('EDUprodcast')
    
    uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
    
    if uploaded_file:
        st.success(f'Uploaded: {uploaded_file.name}')
        
        try:
            with st.spinner('Processing PDF...'):
                index_path, ids_path, chunks_path, summary = process_pdf(uploaded_file)
                
                # Load processed data
                index, ids = load_faiss_index(index_path, ids_path)
                with open(chunks_path, 'r') as f:
                    chunks = json.load(f)
            
            st.success('PDF processed successfully!')
            
            # Create two columns
            col1, col2 = st.columns([0.6, 0.4])
            
            with col1:
                st.markdown("### PDF Preview")
                pdf_display = display_pdf(uploaded_file)
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            with col2:
                # Summary section
                st.markdown("### Document Summary")
                st.markdown(summary)
                
                # Question answering section
                st.markdown("### Ask Questions")
                question = st.text_input("Enter your question about the document:")
                if st.button("Get Answer"):
                    if question:
                        with st.spinner("Generating answer..."):
                            # Find relevant chunks
                            context = search_similar_chunks(question, index, ids, chunks)
                            
                            # Generate answer
                            answer = generate_answer(question, context)
                            
                            st.markdown("#### Answer:")
                            st.markdown(answer)
                    else:
                        st.warning("Please enter a question.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
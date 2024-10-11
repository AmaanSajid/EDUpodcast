import streamlit as st
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os
from unstructured.partition.pdf import partition_pdf
import pytesseract
import uuid
from langchain_community.embeddings import VertexAIEmbeddings
import base64
from google.cloud import aiplatform
from pdfminer.high_level import extract_text
import fitz
from langchain_community.llms.vertexai import VertexAI
from vertexai.language_models import TextGenerationModel
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS as LangchainFAISS
import json
from vertexai.generative_models import GenerativeModel
from typing import List, Tuple
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from io import BytesIO
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
###################################################################

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
aiplatform.init(project="focus-ensign-437302-v1",location="us-central1")

########################################################
###############_________EMBEDDINGS_STUFF_________########################

def extract_elements(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text_elements = []
        table_elements = []
        image_elements = []
        
        for page in doc:
            text_elements.extend([block[4] for block in page.get_text("blocks") if block[6] == 0])
            
            for table in page.find_tables():
                table_elements.append("\n".join(" | ".join(str(cell) for cell in row) for row in table.extract()))
            
            for img in page.get_images(full=True):
                image_data = doc.extract_image(img[0])["image"]
                image_elements.append(base64.b64encode(image_data).decode('utf-8'))
    
    return text_elements, table_elements, image_elements


##EMBEDDING GENERATION

def process_elements(text_elements: List[str], table_elements: List[str], image_elements: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> Tuple[List[str], List[str]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    processed_elements = []
    element_types = []

    for elements, element_type in [(text_elements, 'text'), (table_elements, 'table'), (image_elements, 'image')]:
        for element in elements:
            if element_type in ['text', 'table']:
                chunks = text_splitter.split_text(element)
                processed_elements.extend(chunks)
                element_types.extend([element_type] * len(chunks))
            else:
                processed_elements.append(element)
                element_types.append(element_type)
    
    return processed_elements, element_types

def get_multimodal_embedding(content, content_type):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    
    if content_type in ["text", "table"]:
        embeddings = model.get_embeddings(contextual_text=content, dimension=1408)
        return embeddings.text_embedding
    elif content_type == "image":
        image = Image(base64.b64decode(content))
        embeddings = model.get_embeddings(image=image, dimension=1408)
        return embeddings.image_embedding
    else:
        raise ValueError("Invalid content type")

def create_embeddings(processed_elements: List[str], element_types: List[str]) -> List[np.ndarray]:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_multimodal_embedding, element, element_type) 
                   for element, element_type in zip(processed_elements, element_types)]
        embeddings = [future.result() for future in as_completed(futures)]
    return embeddings

def create_and_save_faiss_index(embeddings: List[np.ndarray], ids: List[int], index_path: str, ids_path: str):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    faiss.write_index(index, index_path)
    
    with open(ids_path, 'w') as id_file:
        json.dump(ids, id_file)

def load_faiss_index(index_path: str, ids_path: str) -> Tuple[faiss.Index, List[int]]:
    index = faiss.read_index(index_path)
    
    with open(ids_path, 'r') as id_file:
        ids = json.load(id_file)
    
    return index, ids

def save_content(processed_elements: List[str], element_types: List[str], content_file_path: str):
    content = [{"content": element, "type": element_type} for element, element_type in zip(processed_elements, element_types)]
    with open(content_file_path, 'w') as f:
        json.dump(content, f)
        
def fetch_content_by_id(id: int, content_file_path: str) -> str:
    with open(content_file_path, 'r') as f:
        content = json.load(f)
    return content[id]['content']

def search_faiss_index(index, ids, query_vector, k=5):
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = [
        {"id": ids[indices[0][i]], "score": float(1 / (1 + distances[0][i]))}
        for i in range(len(indices[0]))
    ]
    return results

def generate_flashcard(question, context, model):
    prompt = f"""
    Based on the following context from a PDF, focusing on the most recent educational content:

    {context}

    Create a concise and accurate flashcard for the question: "{question}"
    Emphasize the latest or most recent educational information from the PDF.
    Format the response as:
    Question: [Refined question based on the context, emphasizing recency]
    Answer: [Concise and accurate answer to the question, highlighting the most recent educational content]
    """
    generation_config = {
        "max_output_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    try:
        response = model.generate_content(
            prompt, 
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True
        )

        full_response = ""
        for chunk in response:
            full_response += chunk.text

        return full_response
    except Exception as e:
        st.error(f"Error generating flashcard: {str(e)}")
        return None

def create_flashcards(questions, index, ids, content_file_path):
    model = GenerativeModel("gemini-1.5-pro-002")
    flashcards = []

    for question in questions:
        try:
            query_embedding = get_multimodal_embedding(question + " latest education", "text")
            similar_items = search_faiss_index(index, ids, query_embedding, k=5)
            
            # Prioritize content from the latter half of the document
            sorted_items = sorted(similar_items, key=lambda x: x['id'], reverse=True)
            context = " ".join(fetch_content_by_id(item['id'], content_file_path) for item in sorted_items[:3])
            
            flashcard = generate_flashcard(question, context, model)
            if flashcard:
                flashcards.append(flashcard)
        except Exception as e:
            st.error(f"Error processing question '{question}': {str(e)}")

    return flashcards

@st.cache_data
def process_pdf(uploaded_file):
    text_elements, table_elements, image_elements = extract_elements(uploaded_file)
    processed_elements, element_types = process_elements(text_elements, table_elements, image_elements, chunk_size=500, chunk_overlap=50)
    embeddings = create_embeddings(processed_elements, element_types)
    
    index_path = 'faiss_index.bin'
    ids_path = 'faiss_ids.json'
    content_file_path = 'content.json'
    
    create_and_save_faiss_index(embeddings, list(range(len(processed_elements))), index_path, ids_path)
    save_content(processed_elements, element_types, content_file_path)
    
    return index_path, ids_path, content_file_path

#################################################################


def main():
    st.title('EDpodcast')
    uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
    
    if uploaded_file is not None:
        st.success(f'Uploaded file: {uploaded_file.name}')
        
        try:
            with st.spinner('Processing PDF...'):
                index_path, ids_path, content_file_path = process_pdf(uploaded_file)
            
            st.success('PDF processed successfully!')

            # Load the FAISS index
            index, ids = load_faiss_index(index_path, ids_path)

            # Flashcard generation
            st.subheader("Generate Flashcards")
            questions = st.text_area("Enter questions for flashcards (one per line):")
            if st.button("Generate Flashcards"):
                question_list = [q.strip() for q in questions.split('\n') if q.strip()]
                if question_list:
                    with st.spinner("Generating flashcards..."):
                        flashcards = create_flashcards(question_list, index, ids, content_file_path)
                    
                    if flashcards:
                        for i, flashcard in enumerate(flashcards, 1):
                            st.write(f"Flashcard {i}:")
                            st.write(flashcard)
                    else:
                        st.warning("No flashcards were generated. Please try different questions or check for errors.")
                else:
                    st.warning("Please enter at least one question.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
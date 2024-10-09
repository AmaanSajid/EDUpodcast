import streamlit as st
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os
from unstructured.partition.pdf import partition_pdf
import pytesseract
import uuid
from langchain_community.embeddings import VertexAIEmbeddings
# from langchain.vectorstores import Chroma
import base64
from google.cloud import aiplatform
# from google.cloud import texttospeech
from pdfminer.high_level import extract_text
import fitz
from langchain_community.llms.vertexai import VertexAI
from vertexai.language_models import TextGenerationModel
# from google.cloud.aiplatform.language_models import TextGenerationModel as _LanguageModel
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS as LangchainFAISS
import json
# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
aiplatform.init(project="focus-ensign-437302-v1",location="us-central1")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# from pdf2image import convert_from_path

poppler_path = r"C:\Users\Amaan Sajid\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin"

### PDF PREPROCESSING AND ELEMENT EXTRACTION
import fitz  # PyMuPDF
import os
import base64
from io import BytesIO
from PIL import Image
from vertexai.vision_models import MultiModalEmbeddingModel





def extract_elements(pdf_file):
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())
    
    output_path = "figures"
    os.makedirs(output_path, exist_ok=True)
    
    doc = fitz.open("temp.pdf")
    text_elements = []
    table_elements = []
    image_elements = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text("blocks")
        for block in text:
            if block[6] == 0:  # Text block
                text_elements.append(block[4])
            elif block[6] == 1:  # Image block
                continue  # We'll handle images separately
        
        # Extract tables
        tables = page.find_tables()
        for table in tables:
            table_text = []
            for row in table.extract():
                table_text.append(" | ".join(str(cell) for cell in row))
            table_elements.append("\n".join(table_text))
        
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            
            # Save image to file
            image_filename = f"image_page{page_num+1}_{img_index+1}.png"
            image_path = os.path.join(output_path, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            
            # Encode image to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            image_elements.append(encoded_image)
    
    doc.close()
    os.remove("temp.pdf")
    
    return text_elements, table_elements, image_elements


##EMBEDDING GENERATION
def get_multimodal_embedding(content, content_type):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    
    if content_type == "text":
        embeddings = model.get_embeddings(
            contextual_text=content,
            dimension=1408
        )
        return embeddings.text_embedding
    elif content_type == "image":
        image = Image(content)
        embeddings = model.get_embeddings(
            image=image,
            dimension=1408
        )
        return embeddings.image_embedding
    else:
        raise ValueError("Invalid content type")

def create_embeddings(text_elements, table_elements, image_elements):
    embeddings = []
    for text in text_elements + table_elements:
        embeddings.append(get_multimodal_embedding(text, "text"))
    for image in image_elements:
        embeddings.append(get_multimodal_embedding(image, "image"))
    return embeddings

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_and_save_faiss_index(embeddings, ids, index_path, ids_path):
    # Create the FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Save the FAISS index
    faiss.write_index(index, index_path)
    
    # Save the IDs
    with open(ids_path, 'w') as id_file:
        json.dump(ids, id_file)

def load_faiss_index(index_path, ids_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load the IDs
    with open(ids_path, 'r') as id_file:
        ids = json.load(id_file)
    
    return index, ids


# def generate_flashcards(text_elements, table_elements):
#     flashcards = []
#     for text in text_elements:
#         flashcards.append({'question': 'What is the key point of this text?', 'answer': text})
#     for table in table_elements:
#         flashcards.append({'question': 'What information does this table represent?', 'answer': table})
#     return flashcards

# def create_summary(text_elements, table_elements):
#     summary = "Summary of the document:\n\n"
#     for i, text in enumerate(text_elements, 1):
#         summary += f"{i}. {text[:100]}...\n"
#     for i, table in enumerate(table_elements, 1):
#         summary += f"Table {i}: {table[:100]}...\n"
#     return summary

# from google.cloud import texttospeech

# def text_to_speech(text, output_filename):
#     client = texttospeech.TextToSpeechClient()
#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.MP3
#     )
#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )
#     with open(output_filename, "wb") as out:
#         out.write(response.audio_content)
#     return output_filename



def main():
    st.title('EDUpodcast')
    uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
    
    if uploaded_file is not None:
        st.success(f'Uploaded file: {uploaded_file.name}')
        
        with st.spinner('Processing PDF...'):
            text_elements, table_elements, image_elements = extract_elements(uploaded_file)
        
        st.write(f"Extracted {len(text_elements)} text elements, {len(table_elements)} tables, and {len(image_elements)} images.")
        
        with st.spinner('Creating embeddings...'):
            all_elements = text_elements + table_elements + image_elements
            embeddings = []
            ids = []
            for i, element in enumerate(all_elements):
                if isinstance(element, str):  # Text or table
                    embedding = get_multimodal_embedding(element, "text")
                else:  # Image
                    embedding = get_multimodal_embedding(element, "image")
                embeddings.append(embedding)
                ids.append(str(i))
        
        # Save the FAISS index
        index_path = 'faiss_index.bin'
        ids_path = 'faiss_ids.json'
        create_and_save_faiss_index(embeddings, ids, index_path, ids_path)
        
        st.success('Embeddings created and stored successfully!')
        
        # # Generate flashcards
        # flashcards = generate_flashcards(text_elements, table_elements)
        # st.subheader("Flashcards")
        # for i, card in enumerate(flashcards[:5], 1):  # Display first 5 flashcards
        #     st.write(f"Flashcard {i}:")
        #     st.write(f"Q: {card['question']}")
        #     st.write(f"A: {card['answer'][:100]}...")  # Display first 100 characters of the answer
        
        # # Create summary for podcast
        # summary = create_summary(text_elements, table_elements)
        # st.subheader("Podcast Summary")
        # st.text(summary)
        
        # # Generate podcast audio
        # if st.button("Generate Podcast"):
        #     with st.spinner("Generating podcast audio..."):
        #         audio_file = text_to_speech(summary, "podcast.mp3")
        #     st.audio(audio_file)
        #     st.success("Podcast generated successfully!")
        
        # # Example of retrieving similar documents
        # if st.button("Find Similar Documents"):
        #     query = st.text_input("Enter a query to find similar documents:")
        #     if query:
        #         # Load the FAISS index
        #         index, stored_ids = load_faiss_index(index_path, ids_path)
                
        #         # Get the query embedding
        #         query_embedding = get_multimodal_embedding(query, "text")
                
        #         # Perform the search
        #         k = 3  # Number of similar documents to retrieve
        #         D, I = index.search(np.array([query_embedding]), k)
                
        #         st.write("Similar documents:")
        #         for i in I[0]:
        #             element_index = int(stored_ids[i])
        #             element = all_elements[element_index]
        #             if isinstance(element, str):
        #                 st.write(element[:200] + "...")  # Display first 200 characters
        #             else:
        #                 st.image(element)  # Display image

if __name__ == '__main__':
    main()


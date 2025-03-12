import streamlit as st
import getpass
import os
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get('OPENAI_API_KEY'): 
    os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

from openai import OpenAI

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, Docx2txtLoader, DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_iris import IRISVector

import os
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document  # For reading .docx files

# # Function to extract text from a Word document
# def load_word_document(file_path):
#     doc = Document(file_path)
#     text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
#     return text

# import pandas as pd

client = OpenAI()
embeddings = OpenAIEmbeddings()

def get_response(main_db, main_embeddings, main_scenario):
    f = open('data/knowledge.docx', 'r', encoding='ISO-8859-1')
    # query = "new technology"
    knowledge = f.read()
        
    embedding_vector = main_embeddings.embed_query(main_scenario)
    res = main_db.similarity_search_by_vector(embedding_vector)

    # # Retrieve top 5 most similar results
    # results = main_db.query(
    #     query_embeddings=[embedding_vector],  # Query embedding
    #     n_results=5  # Number of similar documents to retrieve
    # )

    full_res = ''
    for each_res in res:
        full_res = full_res + '\n\n' +each_res.page_content
            
    # full_res = ''
    # for each_res in results['documents'][0]:
    #     full_res = full_res + '\n\n' +each_res
                
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": 
                    f"""
                    A medical doctor with domain knowledge in breast cancer after having trained with a wealth of knowledge in these topics: {knowledge}.
                    Augment your data with results from {full_res}.
                    """
            },
            {
                "role": "user",
                "content": 
                    f"""
                    Given patient's consultation with the doctor in this {main_scenario}, 
                    1. compare a few potential treatments
                    2. choose a final best recommendation
                    3. provide justifications for the choice
                    
                    Keep answer in short sentences.
                    Detect the language and if it is in Chinese, reply in Chinese
                    """
            }
        ]
    )
    
    response = completion.choices[0].message.content
    
    return response
        
text_response = ''
username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f'iris://{username}:{password}@{hostname}:{port}/{namespace}'
# Under the hood, this becomes a SQL table. CANNOT have '.' in the name

COLLECTION_NAME = 'cancer_db'
# Subsequent calls to reconnect to the database and make searches should use this.  
db = IRISVector(
    embedding_function=embeddings,
    dimension=1536,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
# @st.cache_resource
# def get_data():
#     COLLECTION_NAME = "cancer_db"
#     # Directory containing multiple Word documents
#     folder_path = "data"  # Change this to your actual folder path

#     # List all .docx files in the folder
#     word_files = [f for f in os.listdir(folder_path) if f.endswith(".docx") and 'knowledge' not in f]

#     # # Initialize ChromaDB client
#     chromadb_client = chromadb.Client(path="./chroma_db")#PersistentClient
#     db = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

#     # Text splitter for chunking documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)

#     # Process each Word document
#     doc_id = 0
#     for file_name in word_files:
#         file_path = os.path.join(folder_path, file_name)
#         print(f"Processing: {file_name}")
        
#         # Load and split text
#         document_text = load_word_document(file_path)
#         doc_chunks = text_splitter.split_text(document_text)
        
#         # Generate embeddings
#         actual_embeddings = embeddings.embed_documents(doc_chunks)

#         # Add to ChromaDB
#         db.add(
#             ids=[f"{doc_id}_{i}" for i in range(len(doc_chunks))],  # Unique IDs
#             documents=doc_chunks,  # Text chunks
#             embeddings=actual_embeddings  # Corresponding embeddings
#         )

#         doc_id += 1

#     # print(f"Successfully added {len(word_files)} documents to ChromaDB!")

#     treatment_selection = pd.read_csv('data/treatment_selection.csv')
#     treatment_selection['content'] = treatment_selection['surgery_type'] + ' ' + treatment_selection['benefit'] + ' ' + treatment_selection['consideration'] + ' ' + treatment_selection['tag']
#     texts = treatment_selection['content'].dropna().tolist()  # Remove NaN values and convert to a list
        
#     COLLECTION_NAME = "pictures_db"

#     db1 = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

#     # Text splitter for chunking documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#     # Process text into chunks and embeddings
#     doc_id = 0
#     for text in texts:
#         chunks = text_splitter.split_text(text)  # Split text into smaller chunks
#         actual_embeddings = embeddings.embed_documents(chunks)  # Generate embeddings
        
#         # Add chunks to ChromaDB
#         db1.add(
#             ids=[f"{doc_id}_{i}" for i in range(len(chunks))],  # Unique IDs
#             documents=chunks,  # Text chunks
#             embeddings=actual_embeddings  # Corresponding embeddings
#         )
        
#         doc_id += 1
#     return db, db1
        
# db, db1 = get_data()

st.title('Voice2Choice Beta')

option = st.sidebar.selectbox(
    'Audio or Text?',
    ('Audio', 'Text'),
)

st.write("You selected:", option)

if option == 'Audio':
    audio_value = st.audio_input('Record the consultation dialogue.')

    if audio_value:
        recording = st.audio(audio_value)
        
        st.subheader('Recommendation:')
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_value
        )
        scenario = transcription.text
        st.subheader('Consultation Scenario:')
        st.write(scenario)
        
        st.subheader('Recommendation:')
        text_response = get_response(db, embeddings, scenario)
        st.write(text_response)
    
elif option == 'Text':
    with st.form("my_form"):
        scenario = st.text_area(
            "Enter Patient Consultation Dialogue:"
        )
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            text_response = get_response(db, embeddings, scenario)
            st.write(text_response)            

if text_response:
    st.markdown('You can view the treatment process here.')

    loader = CSVLoader('data/treatment_selection.csv')#, csv_args={'fieldnames':['']})
    docs = loader.load()
    len(docs)

    COLLECTION_NAME = "pictures_db"
    # This creates a persistent vector store (a SQL table). You should run this ONCE only
    db1 = IRISVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    # print(f"Successfully added {len(texts)} rows (split into chunks) to ChromaDB!")    
    
    docs_with_score = db1.similarity_search_with_score(text_response, 1)
    image_chosen = docs_with_score[0][0].page_content.split('\n')[-1].split(': ')[-1] + ".jpeg"
    
    # # Perform a similarity search
    # query_embedding = embeddings.embed_query(text_response)  # Generate embedding for the query

    # # Retrieve top 5 most similar results
    # results = db1.query(
    #     query_embeddings=[query_embedding],  # Query embedding
    #     n_results=1  # Number of similar documents to retrieve
    # )
        
    # image_chosen = results["documents"][0][0].split(' ')[-1] + ".jpeg"        
    
    st.image(f'data/{image_chosen}')
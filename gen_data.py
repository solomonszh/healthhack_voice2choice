import getpass
import pprint
import os

from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

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

import pandas as pd

client = OpenAI()
embeddings = OpenAIEmbeddings()

# Function to extract text from a Word document
def load_word_document(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text

COLLECTION_NAME = "cancer_db"
# Directory containing multiple Word documents
folder_path = "data"  # Change this to your actual folder path

# List all .docx files in the folder
word_files = [f for f in os.listdir(folder_path) if f.endswith(".docx") and 'knowledge' not in f]

# Initialize ChromaDB client
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
db = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)

# Process each Word document
doc_id = 0
for file_name in word_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing: {file_name}")

    # Load and split text
    document_text = load_word_document(file_path)
    doc_chunks = text_splitter.split_text(document_text)

    # Generate embeddings
    actual_embeddings = embeddings.embed_documents(doc_chunks)

    # Add to ChromaDB
    db.add(
        ids=[f"{doc_id}_{i}" for i in range(len(doc_chunks))],  # Unique IDs
        documents=doc_chunks,  # Text chunks
        embeddings=actual_embeddings  # Corresponding embeddings
    )

    doc_id += 1

print(f"Successfully added {len(word_files)} documents to ChromaDB!")

treatment_selection = pd.read_csv('data/treatment_selection.csv')
treatment_selection['content'] = treatment_selection['surgery_type'] + ' ' + treatment_selection['benefit'] + ' ' + treatment_selection['consideration'] + ' ' + treatment_selection['tag']
texts = treatment_selection['content'].dropna().tolist()  # Remove NaN values and convert to a list

COLLECTION_NAME = "pictures_db"

db1 = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Process text into chunks and embeddings
doc_id = 0
for text in texts:
    chunks = text_splitter.split_text(text)  # Split text into smaller chunks
    actual_embeddings = embeddings.embed_documents(chunks)  # Generate embeddings

    # Add chunks to ChromaDB
    db1.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],  # Unique IDs
        documents=chunks,  # Text chunks
        embeddings=actual_embeddings  # Corresponding embeddings
    )

    doc_id += 1

print(f"Successfully added {len(texts)} rows (split into chunks) to ChromaDB!")

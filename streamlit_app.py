import streamlit as st
import getpass
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

client = OpenAI()
embeddings = OpenAIEmbeddings()

response = ''
username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
# Under the hood, this becomes a SQL table. CANNOT have '.' in the name

COLLECTION_NAME = "cancer_db"
# Subsequent calls to reconnect to the database and make searches should use this.  
db = IRISVector(
    embedding_function=embeddings,
    dimension=1536,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

f = open("data/knowledge.docx", "r", encoding='ISO-8859-1')
# query = "new technology"
knowledge = f.read()

st.title("Cancerot Beta")

with st.form("my_form"):
    scenario = st.text_area(
        "Enter Patient Consultation Dialogue:"
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        embedding_vector = embeddings.embed_query(scenario)
        res = db.similarity_search_by_vector(embedding_vector)

        full_res = ''
        for each_res in res:
            full_res = full_res + '\n\n' +each_res.page_content
            
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
                        Given patient's consultation with the doctor in this {scenario}, 
                        1. compare a few potential treatments
                        2. choose a final best recommendation
                        3. provide justifications for the choice
                        
                        Keep answer in short sentences.
                        """
                }
            ]
        )
        
        response = completion.choices[0].message.content
        st.write(response)
        
if response:
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

    docs_with_score = db1.similarity_search_with_score(completion.choices[0].message.content, 1)
    image_chosen = docs_with_score[0][0].page_content.split('\n')[-1].split(': ')[-1] + ".jpeg"
    st.image(f'data/{image_chosen}')
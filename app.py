import streamlit as st
import getpass
import time
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

import pandas as pd

client = OpenAI()
embeddings = OpenAIEmbeddings()

def get_response(main_db, main_embeddings, main_scenario):
    f = open('data/knowledge.docx', 'r', encoding='ISO-8859-1')
    # query = "new technology"
    knowledge = f.read()

    embedding_vector = main_embeddings.embed_query(main_scenario)
    # res = main_db.similarity_search_by_vector(embedding_vector)

    # Retrieve top 5 most similar results
    results = main_db.query(
        query_embeddings=[embedding_vector],  # Query embedding
        n_results=5  # Number of similar documents to retrieve
    )

    # full_res = ''
    # for each_res in res:
    #     full_res = full_res + '\n\n' +each_res.page_content

    full_res = ''
    for each_res in results['documents'][0]:
        full_res = full_res + '\n\n' +each_res

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
                        Detect and Reply in the same language so that for example, if it is in Chinese, reply in Chinese
                        """
                }
            ]
        )

    response = completion.choices[0].message.content

    return response

def get_language(main_scenario):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":
                    f"""
                    Language Detector
                    """
            },
            {
                "role": "user",
                "content":
                    f"""
                    Returns the language used in {main_scenario},
                    """
            }
        ]
    )

    response = completion.choices[0].message.content

    return response

def stream_data(data_to_be_stream):
    for word in data_to_be_stream.split(" "):
        yield word + " "
        time.sleep(0.35)

def iso639_lookup(lang: str, reverse: bool = None, **junk) -> str:
    """
    OpenAI whisper ISO-639-1 language code utility or compatibility - 2024-02

    :param lang: The language name or ISO-639-1 code to look up.
    :param reverse: If True, find the language name from the ISO-639-1 code.
                    If False or None, find the ISO-639-1 code from the language name.
    :return: The ISO-639-1 code or language name if found, otherwise None.
    """
    iso639 = {  # 57 languages supported by OpenAI whisper-1
    'afrikaans': 'af', 'arabic': 'ar', 'armenian': 'hy',
    'azerbaijani': 'az', 'belarusian': 'be', 'bosnian': 'bs',
    'bulgarian': 'bg', 'catalan': 'ca', 'chinese': 'zh',
    'croatian': 'hr', 'czech': 'cs', 'danish': 'da',
    'dutch': 'nl', 'english': 'en', 'estonian': 'et',
    'finnish': 'fi', 'french': 'fr', 'galician': 'gl',
    'german': 'de', 'greek': 'el', 'hebrew': 'he',
    'hindi': 'hi', 'hungarian': 'hu', 'icelandic': 'is',
    'indonesian': 'id', 'italian': 'it', 'japanese': 'ja',
    'kannada': 'kn', 'kazakh': 'kk', 'korean': 'ko',
    'latvian': 'lv', 'lithuanian': 'lt', 'macedonian': 'mk',
    'malay': 'ms', 'maori': 'mi', 'marathi': 'mr',
    'nepali': 'ne', 'norwegian': 'no', 'persian': 'fa',
    'polish': 'pl', 'portuguese': 'pt', 'romanian': 'ro',
    'russian': 'ru', 'serbian': 'sr', 'slovak': 'sk',
    'slovenian': 'sl', 'spanish': 'es', 'swahili': 'sw',
    'swedish': 'sv', 'tagalog': 'tl', 'tamil': 'ta',
    'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk',
    'urdu': 'ur', 'vietnamese': 'vi', 'welsh': 'cy'
    }
    if reverse:
        if len(lang) != 2 or not lang.isalpha():
            raise ValueError("ISO-639-1 abbreviation must be len=2 letters")
        # Find the dict key by searching for the value
        for language, abbreviation in iso639.items():
            if abbreviation == lang.strip().lower():
                return language
        return None  # None if the code not found
    else:
        # match input style to dict format, retrieve
        formatted_lang = lang.strip().lower()
        return iso639.get(formatted_lang)  # will be None for unmatched

text_response = ''
language = 'English'
# Initialize ChromaDB client
chromadb_client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "cancer_db"
db = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

COLLECTION_NAME = "pictures_db"
db1 = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

st.title('Voice2Choice Beta')

option = st.sidebar.selectbox(
    'Video or Audio or Text?',
    ('Audio', 'Video', 'Text'),
)

st.write("You selected:", option)

if option == 'Audio':
    chosen_language = st.selectbox('Select Language:', ('English', 'Chinese', 'Japanese', 'Korean', 'Malay', 'Tamil'))
    language_code = iso639_lookup(chosen_language)

    audio_value = st.audio_input('Record the consultation dialogue.')

    if audio_value:
        recording = st.audio(audio_value)

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_value,
            language=language_code
        )
        scenario = transcription.text
        # language = get_language(scenario)
        # st.write(language)
        st.subheader('Consultation Scenario:')
        # st.write(scenario)
        st.write_stream(stream_data(scenario))

        st.subheader('Recommendation:')
        text_response = get_response(db, embeddings, scenario)
        st.write(text_response)

elif option == 'Video':
    chosen_language = st.selectbox('Select Language:', ('English', 'Chinese', 'Japanese', 'Korean', 'Malay', 'Tamil'))
    language_code = iso639_lookup(chosen_language)

    video_file = open('data/consultation.mp4', 'rb')
    video_bytes = video_file.read()

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=video_file,
        language=language_code
    )
    scenario = transcription.text

    st.video(video_bytes)

    # language = get_language(scenario)
    # st.write(language)
    st.subheader('Consultation Scenario:')
    # st.write(scenario)
    st.write_stream(stream_data(scenario))

    st.subheader('Recommendation:')
    text_response = get_response(db, embeddings, scenario)
    st.write(text_response)

elif option == 'Text':
    with st.form("my_form"):
        scenario = st.text_area(
            "Enter Patient Consultation Dialogue:"
        )
        submitted = st.form_submit_button("Submit")
        language = get_language(submitted)
        if submitted:
            text_response = get_response(db, embeddings, scenario)
            st.write(text_response)

if text_response:
    st.markdown('You can view the treatment process here.')

    # docs_with_score = db1.similarity_search_with_score(text_response, 1)
    # image_chosen = docs_with_score[0][0].page_content.split('\n')[-1].split(': ')[-1] + ".jpeg"

    # Perform a similarity search
    query_embedding = embeddings.embed_query(text_response)  # Generate embedding for the query

    # Retrieve top 5 most similar results
    results = db1.query(
        query_embeddings=[query_embedding],  # Query embedding
        n_results=1  # Number of similar documents to retrieve
    )

    image_chosen = results["documents"][0][0].split(' ')[-1] + ".jpeg"

    st.download_button(
        label='Download Consultation Dialogue and Recommendation Report',
        data=f'Based on {scenario}, \n\n, the Recommendation generally is {text_response}',
        file_name='recommendation.txt',
        # on_click='ignore',
        # type='primary',
        icon=':material/download:',
    )

    st.image(f'data/{image_chosen}')

    with open(f'data/{image_chosen}', 'rb') as file:
        st.download_button(
            label='Download Image',
            data=file,
            file_name='recommendation.jpeg',
            icon=':material/download:',
            # mime='image/png',
        )

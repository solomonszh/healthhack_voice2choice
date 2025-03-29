__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_iris import IRISVector

import os
import chromadb
from docx import Document  # For reading .docx files

import pandas as pd

from pydantic import BaseModel

class Options(BaseModel):
    option1: str
    option2: str
    option3: str

client = OpenAI()
embeddings = OpenAIEmbeddings()

def get_request(main_scenario):
    completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content":
                        f"""
                        Answer Yes or No if related to breast cancer.
                        """
                },
                {
                    "role": "user",
                    "content":
                        f"""
                        Is this about breast cancer? {main_scenario}
                        """
                }
            ]
        )

    response = completion.choices[0].message.content.lower()

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
                    Returns a single word language used in {main_scenario},
                    """
            }
        ]
    )

    response = completion.choices[0].message.content

    return response

def get_response(main_db, main_embeddings, main_scenario, mini_scenario=None, final_language='English'):
    f = open('data/knowledge.docx', 'r', encoding='ISO-8859-1')
    # query = "new technology"
    knowledge = f.read()

    if mini_scenario:
        completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                            f"""
                            A medical doctor with domain knowledge in breast cancer after having trained with a wealth of knowledge in these topics: {knowledge}.
                            """
                    },
                    {
                        "role": "user",
                        "content":
                            f"""
                            1. Decide whether speaker 1 who said {main_scenario} or speaker 2 who said {mini_scenario} is the main patient
                            2. Ignore the non-patient speaker and only focus on the main patient speaker
                            3. Summarize the main patient speaker
                            """
                    }
                ]
            )

        main_scenario = completion.choices[0].message.content

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
                        Reply in {final_language}
                        """
                }
            ]
        )

    response = completion.choices[0].message.content

    if mini_scenario:
        return response, main_scenario
    else:
        return response

def stream_data(data_to_be_stream):
    for word in data_to_be_stream.split(" "):
        yield word + " "
        time.sleep(0.3)

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

# Function to extract text from a Word document
def load_word_document(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text

@st.cache_resource
def generate():
    COLLECTION_NAME = "cancer_db"
    # Directory containing multiple Word documents
    folder_path = "data"  # Change this to your actual folder path

    # List all .docx files in the folder
    word_files = [f for f in os.listdir(folder_path) if f.endswith(".docx") and ('knowledge' not in f) and ('content' not in f)]

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

    return db, db1

@st.cache_data
def get_new_options(convo):
    f = open('data/knowledge.docx', 'r', encoding='ISO-8859-1')
    # query = "new technology"
    knowledge = f.read()

    f = open('data/academia.docx', 'r', encoding='ISO-8859-1')
    # query = "new technology"
    academia = f.read()

    completion = client.beta.chat.completions.parse(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content":
                    f"""
                    A breast cancer doctor trained deeply in {knowledge} and {academia}.
                    """
            },
            {
                "role": "user",
                "content":
                    f"""
                    Based on the latest conversation content below, give 3 follow-up questions from patient's perspectives to aid patient:
                    {convo}
                    """
            }
        ],
        response_format=Options
    )
    response = completion.choices[0].message.parsed
    return [response.option1, response.option2, response.option3]

if "reset" not in st.session_state:
    st.session_state.reset = False

st.title('Voice2Choice Beta')

text_response = ''
language = 'English'
avail_options = ['How are you feeling today?', 'How can I help you today?', 'Do you need something?']
db, db1 = generate()

option = st.sidebar.selectbox(
    'Video or Audio or Text?',
    ('Audio', 'Video', 'Textual Dialogue/Diagnosis', 'Textual Chat'),
)

st.write("You selected:", option)

if option == 'Audio':
    while True:
        chosen_language = st.selectbox('Select Language:', ('English', 'Chinese', 'Japanese', 'Korean', 'Malay', 'Tamil'))
        language_code = iso639_lookup(chosen_language)

        audio_value = st.audio_input('Record the consultation dialogue.')

        if audio_value:
            recording = st.audio(audio_value)

            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_value,
                language=language_code,
                response_format="verbose_json",
                # timestamp_granularities=["segment"]
            )
            # speaker1 = ''
            # speaker2 = ''
            # count=1
            # st.subheader('Consultation Scenario:')
            # for segment in transcription.segments:
            #     st.write_stream(stream_data(f'Speaker{count}: '+segment.text))
            #     if count==1:
            #         speaker1+=segment.text
            #         speaker1+='\n\n'
            #         count=2
            #     else:
            #         speaker2+=segment.text
            #         speaker2+='\n\n'
            #         count=1
            scenario = transcription.text
            # language = get_language(scenario)
            # st.write(language)
            st.subheader('Consultation Scenario:')
            st.write_stream(stream_data(scenario))

            yes_or_no = get_request(scenario)
            st.subheader('Recommendation:')
            if yes_or_no == 'yes':
                text_response = get_response(db, embeddings, scenario, chosen_language)
                # text_response, main_scenario = get_response(db, embeddings, speaker1, speaker2, chosen_language)
                # st.write(main_scenario)
                st.write(text_response)
            else:
                completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content":
                                    f"""
                                    Inform user that more information is needed to provide analysis in {chosen_language}.
                                    """
                            },
                        ]
                    )
                st.write(completion.choices[0].message.content)
        break

elif option == 'Video':
    chosen_language = st.selectbox('Select Language:', ('English', 'Chinese', 'Japanese', 'Korean', 'Malay', 'Tamil'))
    language_code = iso639_lookup(chosen_language)

    video_file = open('data/consultation.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=video_file,
        language=language_code,
        response_format="verbose_json",
        # timestamp_granularities=["segment"]

    )
    # speaker1 = ''
    # speaker2 = ''
    # count=1
    # st.subheader('Consultation Scenario:')
    # for segment in transcription.segments:
    #     st.write_stream(stream_data(f'Speaker{count}: '+segment.text))
    #     if count==1:
    #         speaker1+=segment.text
    #         speaker1+='\n\n'
    #         count=2
    #     else:
    #         speaker2+=segment.text
    #         speaker2+='\n\n'
    #         count=1
    scenario = transcription.text
    # language = get_language(scenario)
    # st.write(language)
    st.subheader('Consultation Scenario:')
    st.write_stream(stream_data(scenario))

    st.download_button(
        label='Download Consultation Dialogue and Recommendation Report',
        data=f'Based on {scenario}, \n\n, the Recommendation generally is {text_response}',
        file_name='recommendation.txt',
        # on_click='ignore',
        # type='primary',
        icon=':material/download:',
    )

    st.subheader('Recommendation:')
    text_response = get_response(db, embeddings, scenario, chosen_language)
    # text_response, main_scenario = get_response(db, embeddings, speaker1, speaker2, chosen_language)
    # st.write(main_scenario)
    st.write(text_response)

elif option == 'Textual Dialogue/Diagnosis':
    with st.form("my_form"):
        scenario = st.text_area(
            "Enter Patient Consultation Dialogue:"
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            language = get_language(scenario).capitalize()
            if scenario.lower() == 'hi':
                language = 'English'
            yes_or_no = get_request(scenario)
            st.subheader('Recommendation:')
            if yes_or_no == 'yes':
                text_response = get_response(db, embeddings, scenario, language)
                st.write(text_response)
            else:
                completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content":
                                    f"""
                                    Inform user that more information is needed to provide analysis in {language}.
                                    """
                            },
                        ]
                    )
                st.write(completion.choices[0].message.content)

elif option == 'Textual Chat':
    st.markdown(
                """
            <style>
                .st-emotion-cache-1c7y2kd {
                    flex-direction: row-reverse;
                    text-align: left;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )

    # Initialize session state for prompt selection
    if 'prompt_selection' not in st.session_state:
        st.session_state.prompt_selection = ""

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    latest_convo = ''
    for convo in st.session_state.chat_messages[-2:]:
        latest_convo += convo['content']
        latest_convo += '\n\n'

    avail_options = get_new_options(latest_convo)
    prompt_selection = st.pills("Prompt suggestions", avail_options, selection_mode="single", label_visibility='hidden')

    input_prompt = st.chat_input("Type your message here...")

    # Get the selected prompt only if it has changed
    def get_prompt_selection():
        global prompt_selection
        if prompt_selection == None:
            return None
        elif prompt_selection == st.session_state.prompt_selection:
            return None
        else:
            st.session_state.prompt_selection = prompt_selection

        return prompt_selection

    if prompt_selection := get_prompt_selection():
        st.session_state.chat_messages.append({"role": "user", "content": str(prompt_selection)})

        with st.chat_message("user"):
            st.markdown(str(prompt_selection))

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            print(response, 'here')
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # Trigger a rerun to update the chat messages
        st.rerun()

    if input_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": input_prompt})

        with st.chat_message("user"):
            st.markdown(input_prompt)

        if 'image' in input_prompt.lower():
            # Perform a similarity search
            query_embedding = embeddings.embed_query(input_prompt)  # Generate embedding for the query

            # Retrieve top 5 most similar results
            results = db1.query(
                query_embeddings=[query_embedding],  # Query embedding
                n_results=1  # Number of similar documents to retrieve
            )

            image_chosen = results["documents"][0][0].split(' ')[-1] + ".jpeg"

            st.image(f'data/{image_chosen}')
        else:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)

            st.session_state.chat_messages.append({"role": "assistant", "content": response})

            # Trigger a rerun to update the chat messages
            st.rerun()

if text_response:
    # docs_with_score = db1.similarity_search_with_score(text_response, 1)
    # image_chosen = docs_with_score[0][0].page_content.split('\n')[-1].split(': ')[-1] + ".jpeg"

    st.markdown('You can view the treatment process here.')

    # Perform a similarity search
    query_embedding = embeddings.embed_query(text_response)  # Generate embedding for the query

    # Retrieve top 5 most similar results
    results = db1.query(
        query_embeddings=[query_embedding],  # Query embedding
        n_results=1  # Number of similar documents to retrieve
    )

    image_chosen = results["documents"][0][0].split(' ')[-1] + ".jpeg"

    st.image(f'data/{image_chosen}')

    with open(f'data/{image_chosen}', 'rb') as file:
        st.download_button(
            label='Download Image',
            data=file,
            file_name='recommendation.jpeg',
            icon=':material/download:',
            # mime='image/png',
        )

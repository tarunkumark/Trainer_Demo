import streamlit as st
from PyPDF2 import PdfReader
import psycopg2
from psycopg2.extras import RealDictCursor
from htmlTemplates import css, bot_template, user_template
import requests
import json
import os
import io
from streamlit_session_browser_storage import SessionStorage
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

st.set_page_config(page_title="Netcore GPT", page_icon=":books:")

# Initialize SessionStorage
sessionBrowserS = SessionStorage()

DATABASE_URL = st.secrets["database"]["url"]

def construct_token_dict():
    token_dict = {
        "web": {
            "client_id": st.secrets["token_dict"]["client_id"],
            "project_id": st.secrets["token_dict"]["project_id"],
            "auth_uri": st.secrets["token_dict"]["auth_uri"],
            "token_uri": st.secrets["token_dict"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["token_dict"]["auth_provider_x509_cert_url"],
            "client_secret": st.secrets["token_dict"]["client_secret"],
            "redirect_uris": st.secrets["token_dict"]["redirect_uris"],
        }
    }
    return token_dict

# Database connection
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Function to save a message to the database
def save_message(user_id, message, is_user):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, message, is_user) VALUES (%s, %s, %s)",
        (user_id, message, is_user),
    )
    conn.commit()
    cursor.close()
    conn.close()

# Function to load chat history for a user
def load_chat_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT message, is_user FROM chat_history WHERE user_id = %s ORDER BY timestamp ASC",
        (user_id,),
    )
    chat_history = cursor.fetchall()
    cursor.close()
    conn.close()
    return chat_history

# Function to save credentials to SessionStorage
def save_credentials(credentials):
    credentials_dict = json.loads(credentials.to_json())
    sessionBrowserS.setItem("google_drive_credentials", credentials_dict)

# Function to load credentials from SessionStorage
def load_credentials():
    credentials_dict = sessionBrowserS.getItem("google_drive_credentials")
    print(credentials_dict)
    if credentials_dict:
        try:
            credentials = Credentials.from_authorized_user_info(
                json.loads(credentials_dict)
            )
        except:
            credentials = Credentials.from_authorized_user_info(
                credentials_dict
            )
        return credentials
    return None

def delete_credentials():
    sessionBrowserS.deleteAll()

# Function to authenticate and get the Google Drive service
def get_google_drive_service():
    creds = load_credentials()

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            save_credentials(creds)
        else:
            token_dict = construct_token_dict()
            flow = InstalledAppFlow.from_client_config(
                token_dict,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
                redirect_uri="https://trainer-demo.streamlit.app/",
            )
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                prompt='consent'
            )
            st.write(
                f'<a href="{auth_url}" target="_blank">Authorize Google Drive</a>',
                unsafe_allow_html=True,
            )
            return None

    if creds:
        service = build("drive", "v3", credentials=creds)
        save_credentials(creds)  # Save credentials after refreshing
        return service
    else:
        st.write("Google Drive authorization failed.")
        return None

def list_pdfs(service):
    results = (
        service.files()
        .list(
            q="mimeType='application/pdf'",
            pageSize=10,
            fields="nextPageToken, files(id, name)",
        )
        .execute()
    )
    items = results.get("files", [])
    pdf_files = {item["name"]: item["id"] for item in items}
    return pdf_files

def download_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO(request.execute())
    return file

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    embeddings = OpenAIEmbeddings(api_key=st.secrets["openai"]["api_key"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    llm = ChatOpenAI(api_key=st.secrets["openai"]["api_key"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def handle_userinput(user_id, user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            save_message(user_id, message.content, True)
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            save_message(user_id, message.content, False)
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )

def main():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Netcore GPT :books:")
    user_question = st.text_input("Ask me anything from the documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process Uploaded PDFs"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.subheader("Google Drive PDFs")
        query_params = st.experimental_get_query_params()
        if "code" in query_params:
            flow = InstalledAppFlow.from_client_config(
                construct_token_dict(),
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
                redirect_uri="https://trainer-demo.streamlit.app/",
            )
            flow.fetch_token(code=query_params["code"][0])
            creds = flow.credentials
            st.session_state["token"] = json.loads(creds.to_json())
            save_credentials(creds)
            st.experimental_set_query_params()  # Clear query params after fetching token
        service = get_google_drive_service()
        if service:
            pdf_files = list_pdfs(service)
            selected_pdfs = st.multiselect(
                "Select PDFs to process", options=list(pdf_files.keys())
            )
            if st.button("Process Selected PDFs"):
                with st.spinner("Processing"):
                    pdf_docs = [
                        download_pdf(service, pdf_files[name]) for name in selected_pdfs
                    ]
                    raw_text = get_pdf_text(pdf_docs)
                    st.write(raw_text)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            st.button("Logout", on_click=delete_credentials)

if __name__ == "__main__":
    main()

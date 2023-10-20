import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# Define the login credentials
LOGIN_USERNAME = "pdf-1M"
LOGIN_PASSWORD = "PDF1MXR45"

# Function to authenticate the user
def authenticate(username, password):
    return username == LOGIN_USERNAME and password == LOGIN_PASSWORD

# Function to get PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Streamlit UI
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Check if the user is authenticated
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.header("Dasturga kirish")
        username = st.text_input("Login")
        password = st.text_input("Parol", type="password")
        if st.button("Kirish"):
            if authenticate(username, password):
                st.session_state.authenticated = True
            else:
                st.error("Xato login yoki parol.")

    if st.session_state.authenticated:
        st.header("Bir nechta fayl bilan suhbat :books:")
        user_question = st.text_input("Fayllaringiz asosida savol bering:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Sizning hujjatlaringiz")
            pdf_docs = st.file_uploader(
                "Faylni tanlang va 'Process' tugmasini bosing", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Tahlil qilyapman, iltimos kuting..."):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create a vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Create a conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

if __name__ == '__main__':
    main()

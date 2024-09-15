import os
import streamlit as st
import speech_recognition as sr
import tempfile
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_chat import message
from streamlit_mic_recorder import speech_to_text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import asyncio
from edge_tts import Communicate


# Set page configuration
st.set_page_config(page_title="Urdu Document Assistant", page_icon="💬")


# Extract text from uploaded PDFs
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        text += get_pdf_text(uploaded_file)
    return text

# Read and extract text from a single PDF file
def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into smaller chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# Create vector store for document retrieval
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

# Generate AI response using a retrieval-augmented generation (RAG) chain
def retrieval_chain(vector_db, input_query, google_api_key):
    template = """
    You are an AI assistant that assists users by extracting information from the provided context:
    {context}.
    Greet the user politely. Ask how you can assist them with context-related queries.
    
    You have to strictly follow the following rules:

    - You have to Provide informative and relevant responses to questions about context.
    - If the user asks about a topic unrelated to context, politely say please ask the question from the document.
    - Be patient and considerate when responding to user queries, and provide clear explanations.
    - Respond in simple and proper Urdu language when the user asks a question. Do not use any other language.
    - Avoid using Hindi words.
    - Do not use English language.
    - Avoid using bold or bullet points.
    - Keep responses polite and concise (max 400 words).
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, google_api_key=google_api_key)
    output_parser = StrOutputParser()
    rag_chain = setup_and_retrieval | prompt | model | output_parser
    response = rag_chain.invoke(input_query)
    return response



# Convert text response to voice using edge-tts
async def edge_text_to_speech(text, voice_name='ur-PK-AsadNeural'):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        save_path = temp_audio_file.name

    communicate = Communicate(voice=voice_name, text=text)
    await communicate.save(save_path)

    return save_path

def play_urdu_voice(response_text):
    try:
        audio_path = asyncio.run(edge_text_to_speech(response_text))
        return audio_path
    except Exception as e:
        st.error(f"Error generating voice: {e}")
        return ""

# Display chat history and play audio
def display_chat_history():
    response_container = st.container()
    with response_container:
        for i, message_data in enumerate(reversed(st.session_state.chat_history)):
            if "audio" in message_data and message_data["audio"]:
                st.audio(message_data["audio"], autoplay=not message_data["is_user"])
            message(message_data['content'], is_user=message_data['is_user'], key=str(i))
 

# Store API key 
with st.sidebar:
    st.title("🔑 API Key")
    api_key = st.text_input('google API key (Optional)', key="gemini_key", type="password")
               
default_key = "AIzaSyDdH61P5vBdMLp-wm9z779ASWr85qY5AWw"
# Main app function
def main():
    st.header("📝 Talk to Documents in URDU")
    

    google_api_key = api_key if api_key else default_key  # Get API key from session state

    # if google_api_key:
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    if "conversation" not in st.session_state:
            st.session_state.conversation = None


    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF files (Multiple)", type=['pdf'], accept_multiple_files=True)    
    process = None
    if uploaded_files:
        process = st.sidebar.button("Process", type="primary")
        
        if process:
            with st.spinner('📄 Loading and processing files...'):
                    files_text = get_files_text(uploaded_files)
                    text_chunks = get_text_chunks(files_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.success(" Files processed successfully!", icon="✅")
                    st.session_state.conversation = vectorstore
               
    else:
        st.error("📃 Please upload and process your PDF files in the sidebar.")
        
    
    if st.session_state.conversation:
                
                
        col1, col2 = st.columns([3, 1])
            
            # Voice input section
        with col2:
            # if st.button("🎤 Record ", type= 'primary'):
                with st.spinner("🚀Processing...",):
                    try:
                        text = speech_to_text(language="ur", use_container_width=True, start_prompt="🎤 Start Recording", stop_prompt="Stop Recording", just_once=True, key="STT")
                            

                    except sr.UnknownValueError:
                        st.error("Could not understand the audio.")
                    except sr.RequestError as e:
                                st.error(f"Request Error: {e}")
                
                
                # user_question, user_audio_path = record_and_recognize()
                user_question = text
                if user_question:
                        response_text = retrieval_chain(st.session_state.conversation, user_question, google_api_key)
                        st.session_state.chat_history.append({"content": user_question, "is_user": True})
                        response_audio_path = play_urdu_voice(response_text)
                        st.session_state.chat_history.append({"content": response_text, "is_user": False, "audio": response_audio_path})

            # Text input section
        with col1:
            user_question = st.chat_input("💬 Ask a question about your files" )
            if user_question:
                    response_text = retrieval_chain(st.session_state.conversation, user_question, google_api_key)
                    st.session_state.chat_history.append({"content": user_question, "is_user": True, "audio": None})
                    response_audio_path = play_urdu_voice(response_text)
                    st.session_state.chat_history.append({"content": response_text, "is_user": False, "audio": response_audio_path})

        display_chat_history()

    

# Run the app
if __name__ == '__main__':
    main()
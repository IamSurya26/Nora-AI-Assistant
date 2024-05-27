import os
from dotenv import load_dotenv
import streamlit as st
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
 
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
 
# Initialize the session state history if it doesn't exist
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
 
# Function to get text from PDF files
def get_pdf_text(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                all_text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    return all_text
 
# Function to get text chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)
 
# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store
 
# Handle user input and process the response
def handle_user_input(user_question):
    vector_store = st.session_state.get("vector_store")
   
    if vector_store:
        docs = vector_store.similarity_search(user_question)
        llm = OpenAI(api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
       
        # Process the question and generate a response
        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": user_question, "input_documents": docs})
            ai_response = response["output_text"]
       
        # Update session state with the new conversation
        dialogue = {"user_question": user_question, "ai_response": ai_response}
        st.session_state.history.append(dialogue)
       
        # Refresh the chat history display
        st.experimental_rerun()
    else:
        st.error("Please upload and process PDF files first.")
 
# Main function to run the Streamlit app
def main():
    # Check if logged in
    if not st.session_state.get("logged_in"):
        login_page()
    else:
        main_app()

# Define the login page
def login_page():
      
    # Center the logo within the login section
    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns to center the logo
    with col2:
        st.image("ourlogo.png", width=400)

    st.title("Login")

    login_form = st.form(key="login_form")
    username = login_form.text_input("Username")
    password = login_form.text_input("Password", type="password")
    submit_login = login_form.form_submit_button("Login")

    # Authentication logic (replace with your actual logic)
    if submit_login:
        if username == "admin@NoraAI.io" and password == "Ch@tA$$i$t@nt2O24":
            st.session_state.logged_in = True
        else:
            st.error("Invalid login credentials")

# Define the main app logic
def main_app():
    # Set page configuration and title
    st.set_page_config(page_title="Nora - Your AI Assistant", layout="wide")
    st.title("Nora - Your AI Assistant")
 
    # Define custom styles for questions and answers
    st.markdown("""
        <style>
        .question { color: blue; }
        .answer { color: green; }
        </style>
    """, unsafe_allow_html=True)
 
    # Initialize the session state history
    initialize_session_state()
 
    # Display chat history at the top
    for dialogue in st.session_state.history:
        question_html = f'<p class="question">You: {dialogue["user_question"]}</p>'
        answer_html = f'<p class="answer">Nora AI: {dialogue["ai_response"]}</p>'
        st.markdown(question_html, unsafe_allow_html=True)
        st.markdown(answer_html, unsafe_allow_html=True)
 
    # Sidebar for PDF upload
    st.sidebar.image("ourlogo.png", width=260)
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    processing_message = st.sidebar.empty()
 
    # Handle "Submit & Process" button click
    if st.sidebar.button("Submit & Process"):
        processing_message.text("Processing...")
        raw_text = get_pdf_text(pdf_docs)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state["vector_store"] = vector_store
            processing_message.success("Done")
        else:
            processing_message.error("No PDF files uploaded or an error occurred!")
 
    # Form for user input at the bottom of the page
    with st.form(key="user_input_form"):
        user_question = st.text_input("Ask a question:", key="user_input")
        submit_button = st.form_submit_button("Send")
   
    # Handle form submission
    if submit_button and user_question:
        handle_user_input(user_question)

# Run the main function
if __name__ == "__main__":
    main()

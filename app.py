import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import requests
from bs4 import BeautifulSoup

# Function to extract text from a single PDF (for parallel processing)
def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to extract text from multiple PDFs in parallel
def get_pdf_text(pdf_docs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_docs))
    return " ".join(texts)  # Combine all text from each PDF

# Function to fetch text from a single website URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text from common tags
        for script in soup(["script", "style"]):
            script.decompose()  # Remove script and style elements

        text = soup.get_text(separator="\n")
        return text.strip()
    except Exception as e:
        st.write(f"Error reading URL {url}: {e}")
        return ""

# Function to extract text from multiple URLs in parallel
def get_url_text(urls):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_url, urls))
    return " ".join(texts)  # Combine all text from each URL

# Function to split text into chunks (optimized with a single instance of the splitter)
def get_text_chunks(text):
    # Create a single instance of RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1500,      # Increased chunk size for larger context
        chunk_overlap=100,    # Reduced overlap for efficiency
        length_function=len
    )
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # Embeddings from OpenAI
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversational chain
def get_conversation_chain(vectorstore):
    # Define a custom prompt template
    prompt_template = """
    You are a knowledgeable assistant capable of answering questions, explaining concepts, summarizing content, and generating examples based on the provided context. The context may include text. Follow these instructions:

    - If the user asks a **question related to the content**, provide a concise and accurate answer based only on the provided context.
    - If the user wants to **understand a concept**, provide a clear and detailed explanation.
    - If the user requests a **summary**, provide a concise and comprehensive summary of the relevant content.
    - If the user asks for **additional examples**, create examples based on the concepts or data available in the context.
    - If the user asks a **question outside the provided context** or about **something unknown** in the context, use your broader knowledge (API) to provide an accurate and informative answer.

    Here is the context:

    {context}

    Question: {question}

    Respond accordingly based on the user's request without additional commentary or introductory phrases.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Define the chat model
    llm = ChatOpenAI(temperature=0.5)

    # Define memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Explicitly specify the output key
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

# Main function
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs and URLs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs and URLs :books:")
    user_question = st.text_input("Ask a question about your documents or URLs:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"])
        
        st.subheader("Website URLs")
        urls = st.text_area("Enter URLs (one per line)")

        if st.button("Process"):
            with st.spinner("Processing"):
                # Step 1: Extract text from PDFs
                st.write("Step 1: Extracting text from PDFs...")
                pdf_text = get_pdf_text(pdf_docs) if pdf_docs else ""
                if pdf_text:
                    st.write("✅ Step 1 Complete: Text extracted from PDFs.")
                else:
                    st.write("No PDFs uploaded or no text extracted from PDFs.")

                # Step 2: Extract text from URLs
                st.write("Step 2: Extracting text from URLs...")
                url_list = urls.splitlines()
                url_text = get_url_text(url_list) if urls.strip() else ""
                if url_text:
                    st.write("✅ Step 2 Complete: Text extracted from URLs.")
                else:
                    st.write("No URLs provided or no text extracted from URLs.")

                # Combine all text
                combined_text = pdf_text + " " + url_text

                if not combined_text.strip():
                    st.write("❗ No content available to process. Please upload PDFs or enter valid URLs.")
                    return  # Stop further processing if no content is available

                # Step 3: Split text into chunks
                st.write("Step 3: Splitting text into chunks...")
                text_chunks = get_text_chunks(combined_text)
                st.write("✅ Step 3 Complete: Text split into chunks.")

                # Step 4: Create vector store
                st.write("Step 4: Creating vector store for text embeddings...")
                vectorstore = get_vectorstore(text_chunks)
                st.write("✅ Step 4 Complete: Vector store created.")

                # Step 5: Set up conversation chain
                st.write("Step 5: Setting up conversation chain...")
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.write("✅ Step 5 Complete: Conversation chain ready.")

if __name__ == '__main__':
    main()

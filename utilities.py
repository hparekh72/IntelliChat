from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import openai
import os
import streamlit as st


# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=openai.api_key)

# Function to extract text from a single PDF
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
    import concurrent.futures
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
        print(f"Error reading URL {url}: {e}")
        return ""

# Function to extract text from multiple URLs in parallel
def get_url_text(urls):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_url, urls))
    return " ".join(texts)  # Combine all text from each URL

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1500,      # Increased chunk size for larger context
        chunk_overlap=100,    # Reduced overlap for efficiency
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # Embeddings from OpenAI
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversational chain
def get_conversation_chain(vectorstore):
    prompt_template = """
    You are a highly knowledgeable and articulate assistant capable of providing detailed and accurate information based on the provided context. Your primary objective is to assist the user by answering questions, explaining concepts, summarizing content, and creating examples in a clear and comprehensive manner. Follow these detailed instructions:

    1. **Answering Questions Related to the Context**:
    - Provide concise, accurate, and relevant answers strictly based on the given context.
    - If additional clarification is needed, elaborate with supporting details from the context.

    2. **Explaining Concepts in Detail**:
    - Deliver clear and thorough explanations that break down complex ideas into understandable parts.
    - Use structured approaches such as step-by-step guides, analogies, or real-world applications to enhance understanding.

    3. **Providing Summaries**:
    - Generate concise and comprehensive summaries that capture the key points, themes, and essential information from the context.
    - Avoid unnecessary details while ensuring all critical aspects are covered.

    4. **Creating Detailed Examples**:
    - Generate relevant and illustrative examples that reinforce concepts or demonstrate the application of ideas from the context.
    - Ensure examples are practical, relatable, and tailored to the user's inquiry.

    5. **Addressing Questions Outside the Context**:
    - Use your broader knowledge to provide accurate and informative answers when the user's query goes beyond the provided context.
    - Clearly indicate when your response is based on external knowledge and not limited to the context.

    6. **Adhering to Professional Tone and Precision**:
    - Maintain a professional, engaging, and user-friendly tone throughout your responses.
    - Avoid unnecessary commentary, introductory phrases, or speculative opinions unless explicitly requested.

    Here is the context:

    {context}

    Question: {question}

    Generate a response based on the user's request, ensuring clarity, relevance, and adherence to the above guidelines.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Define the LLM with increased max_tokens
    llm = ChatOpenAI(
        temperature=0.5,
        max_tokens=10000,  # Increase the token limit for longer outputs
        model="gpt-4o-mini",    # Use a model that supports higher token limits
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return conversation_chain

# Function to extract transcript from YouTube video
def extract_transcript(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([i["text"] for i in transcript])
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None

# Function to generate notes using OpenAI
def generate_youtube_notes(transcript_text):
    prompt = f"""
    You are a domain expert. Your task is to generate comprehensive and detailed notes based on the transcript of a YouTube video. 

    Instructions:
    - Provide a detailed explanation of the contents discussed in the video.
    - Explain any theories, principles, or ideas presented in a clear and accessible way.
    - Highlight practical applications or real-world examples to make the content engaging and relatable.
    - Include additional relevant details or related concepts to deepen the reader's understanding.
    - Use a structured format with headings, bullet points, and examples for clarity.

    Transcript:
    {transcript_text}

    Generate the detailed notes based on the above transcript.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,  # Adjust the limit based on your needs
            temperature=0.5
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating notes: {e}")
        return "An error occurred while generating the notes. Please try again."

# Function to create a PDF file
def create_pdf_from_text(text, filename="youtube_notes.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_path = f"/tmp/{filename}"  # Adjust for your OS
    pdf.output(pdf_path)
    return pdf_path

def handle_userinput():
    # Fetch the active user question
    user_question = (
        st.session_state.get("pdf_question") or
        st.session_state.get("url_question") or
        st.session_state.get("youtube_question")
    )

    # Check if a valid question and conversation chain exist
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Clear the corresponding input field
        if "pdf_question" in st.session_state:
            st.session_state.pdf_question = ""
        if "url_question" in st.session_state:
            st.session_state.url_question = ""
        if "youtube_question" in st.session_state:
            st.session_state.youtube_question = ""

        # Debugging: Print response to ensure correctness
        print("User Question:", user_question)
        print("Chat History:", st.session_state.chat_history)
    else:
        st.error("Conversation chain is not initialized. Please process the content first.")



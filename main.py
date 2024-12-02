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
from youtube_transcript_api import YouTubeTranscriptApi
import os
import openai

# Load environment variables
load_dotenv()

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_docs))
    return " ".join(texts)

# Function to fetch text from a single website URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        st.write(f"Error reading URL {url}: {e}")
        return ""

# Function to extract text from multiple URLs
def get_url_text(urls):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text_from_url, urls))
    return " ".join(texts)

# Function to extract transcript from YouTube video
def extract_transcript(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript])
        return transcript_text
    except Exception as e:
        st.write(f"Error extracting transcript: {e}")
        return None


# Function to generate notes using OpenAI GPT-4o-mimi
def generate_notes(transcript_text, subject):
    # Improved prompt
    prompt = f"""
    You are an expert in {subject}. Your task is to generate comprehensive and detailed notes based on the transcript of a YouTube video. 
    
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
        # Call OpenAI's API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use the appropriate GPT-4 model
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=3000  # Adjust based on your token limit
        )

        # Extract and return the response content
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Error generating notes: {e}")
        return "An error occurred while generating the notes. Please try again."


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversational chain
def get_conversation_chain(vectorstore):
    prompt_template = """
    You are a knowledgeable assistant capable of answering questions, explaining concepts, summarizing content, and generating examples based on the provided context. Follow these instructions:

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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Main function
def main():
    st.title("Document, Website, and YouTube Summary Chatbot")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Input Options")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        urls = st.text_area("Enter Website URLs (one per line)")
        youtube_url = st.text_input("Enter YouTube Video URL")

    if st.button("Process"):
        with st.spinner("Processing your inputs..."):
            combined_text = ""

            # Process PDFs
            if pdf_docs:
                st.write("Extracting text from PDFs...")
                pdf_text = get_pdf_text(pdf_docs)
                combined_text += pdf_text
                st.write("✅ PDFs processed successfully!")

            # Process URLs
            if urls.strip():
                st.write("Extracting text from URLs...")
                url_text = get_url_text(urls.splitlines())
                combined_text += url_text
                st.write("✅ URLs processed successfully!")

            # Process YouTube video
            if youtube_url.strip():
                st.write("Extracting transcript from YouTube...")
                transcript = extract_transcript(youtube_url)
                if transcript:
                    st.write("✅ YouTube transcript extracted successfully!")
                    subject = st.selectbox("Select the subject for YouTube summary:", ["General", "Physics", "Chemistry", "Mathematics", "Data Science"])
                    summary = generate_notes(transcript, subject)
                    st.markdown("### YouTube Video Summary")
                    st.write(summary)

            # Combine text
            if combined_text.strip():
                st.write("Splitting text into chunks...")
                text_chunks = get_text_chunks(combined_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.write("✅ Ready for Q&A!")

if __name__ == "__main__":
    main()

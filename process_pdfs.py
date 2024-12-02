import streamlit as st
from utilities import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput
from htmlTemplates import css, user_template, bot_template

def process_pdfs_page():
    st.write(css, unsafe_allow_html=True)  # Use custom CSS

    st.header("Process PDFs :books:")

    # File uploader for PDFs
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"]
    )

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                # Step 1: Extract text from PDFs
                st.write("Extracting text from PDFs...")
                pdf_text = get_pdf_text(pdf_docs)
                if pdf_text:
                    st.write("✅ Text extracted from PDFs.")

                    # Step 2: Split text into chunks
                    st.write("Splitting text into chunks...")
                    text_chunks = get_text_chunks(pdf_text)
                    st.write("✅ Text split into chunks.")

                    # Step 3: Create vector store
                    st.write("Creating vector store for text embeddings...")
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("✅ Vector store created.")

                    # Step 4: Set up conversation chain
                    st.write("Setting up conversation chain...")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.write("✅ Conversation chain ready! You can now ask questions about the PDFs.")
                else:
                    st.error("No text could be extracted from the uploaded PDFs.")
        else:
            st.error("Please upload PDFs to process.")

    # Chat feature
    st.markdown("---")
    st.subheader("Chat with PDFs")
    st.text_input("Ask a question about your PDFs:", key="pdf_question", on_change=handle_userinput)

    # Display chat messages
    if "chat_history" in st.session_state:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


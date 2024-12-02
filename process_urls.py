import streamlit as st
from utilities import get_url_text, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput
from htmlTemplates import css, user_template, bot_template

def process_urls_page():
    st.write(css, unsafe_allow_html=True)  # Use custom CSS

    st.header("Process URLs :link:")

    # Input URLs
    urls = st.text_area("Enter URLs (one per line):")

    if st.button("Process URLs"):
        if urls.strip():
            with st.spinner("Processing URLs..."):
                # Step 1: Extract text from URLs
                st.write("Extracting text from URLs...")
                url_list = urls.splitlines()
                url_text = get_url_text(url_list)
                if url_text:
                    st.write("✅ Text extracted from URLs.")

                    # Step 2: Split text into chunks
                    st.write("Splitting text into chunks...")
                    text_chunks = get_text_chunks(url_text)
                    st.write("✅ Text split into chunks.")

                    # Step 3: Create vector store
                    st.write("Creating vector store for text embeddings...")
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("✅ Vector store created.")

                    # Step 4: Set up conversation chain
                    st.write("Setting up conversation chain...")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.write("✅ Conversation chain ready! You can now ask questions about the URLs.")
                else:
                    st.error("Failed to extract text from the URLs.")
        else:
            st.error("Please enter URLs to process.")

    # Chat feature
    st.markdown("---")
    st.subheader("Chat with URLs")
    st.text_input("Ask a question about your URLs:", key="url_question", on_change=handle_userinput)

    # Display chat messages
    if "chat_history" in st.session_state:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


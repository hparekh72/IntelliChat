import streamlit as st
from utilities import extract_transcript, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput, generate_youtube_notes, create_pdf_from_text
from htmlTemplates import css, user_template, bot_template

def process_youtube_page():
    st.write(css, unsafe_allow_html=True)  # Use custom CSS

    st.header("Process YouTube Videos :movie_camera:")

    # Input YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if st.button("Process YouTube Video"):
        if youtube_link.strip():
            with st.spinner("Processing YouTube video..."):
                # Step 1: Extract Transcript
                st.write("Extracting transcript from YouTube video...")
                youtube_text = extract_transcript(youtube_link)
                if youtube_text:
                    st.write("✅ Transcript extracted successfully!")
                    st.image(
                        f"http://img.youtube.com/vi/{youtube_link.split('=')[-1]}/0.jpg",
                        use_container_width=True,
                    )

                    # Step 2: Generate Notes
                    st.write("Generating notes from YouTube transcript...")
                    notes = generate_youtube_notes(youtube_text)
                    if notes:
                        st.markdown("## Generated Notes:")
                        st.write(notes)

                        # Step 3: Create PDF and Provide Download Button
                        pdf_path = create_pdf_from_text(notes)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Notes as PDF",
                                data=pdf_file,
                                file_name="youtube_notes.pdf",
                                mime="application/pdf"
                            )

                        # Step 4: Set Up Conversation
                        st.write("Setting up Q&A conversation for YouTube video...")
                        
                        # Combine the transcript and notes for chunking
                        combined_text = youtube_text + "\n\n" + notes

                        # Split text into chunks
                        text_chunks = get_text_chunks(combined_text)
                        st.write("✅ Text split into chunks.")

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        st.write("✅ Vector store created.")

                        # Set up conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.write("✅ Conversation chain ready! You can now ask questions about the video.")
                    else:
                        st.error("Failed to generate notes from the YouTube transcript.")
                else:
                    st.error("Failed to extract transcript from the YouTube video.")
        else:
            st.error("Please enter a valid YouTube video link.")

    # Chat feature
    st.markdown("---")
    st.subheader("Chat with YouTube Video")
    st.text_input("Ask a question about your YouTube video:", key="youtube_question", on_change=handle_userinput)

    # Display chat messages
    if "chat_history" in st.session_state:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


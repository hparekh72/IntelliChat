import streamlit as st
from dotenv import load_dotenv
from process_pdfs import process_pdfs_page
from process_urls import process_urls_page
from process_youtube import process_youtube_page

# Load environment variables
load_dotenv()

# Function to initialize session state
def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # Initialize the conversation chain
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize the chat history
    if "pdf_question" not in st.session_state:
        st.session_state.pdf_question = ""  # Initialize PDF question input
    if "url_question" not in st.session_state:
        st.session_state.url_question = ""  # Initialize URL question input
    if "youtube_question" not in st.session_state:
        st.session_state.youtube_question = ""  # Initialize YouTube question input

# Navbar function with custom styling
def render_navbar():
    st.sidebar.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 20px;
        }
        .sidebar-radio {
            font-size: 16px;
            font-weight: bold;
        }
        .sidebar-radio > label {
            font-size: 20px;
            color: #f04e30; /* Highlight active selection */
        }
        .sidebar-radio div[role="radiogroup"] {
            gap: 10px; /* Add spacing between buttons */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to:",
        ["Process PDFs", "Process URLs", "Process YouTube Videos"],
        key="navbar",  # Assign a unique key for the navbar
    )

# Function to render the Start Over button at the bottom
# Function to render the Start Over button at the bottom
def render_start_over_button():
    st.markdown(
        """
        <style>
        .start-over-btn {
            margin-top: 30px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Start Over Button
    with st.container():
        if st.button("Clear Chat", key="start_over"):
            st.session_state.clear()  # Clear all session state variables
            st.session_state["navbar"] = "Process PDFs"  # Set default navigation state
            st.session_state["conversation"] = None  # Reset the conversation chain
            st.query_params.clear()  # Clear query parameters
            st.session_state["rerun"] = True  # Trigger a rerun

# Main function
def main():
    st.set_page_config(
        page_title="Chat with PDFs, URLs, and YouTube Videos",
        page_icon=":books:",
        layout="wide",  # Use the full-width layout for better design
    )
    
    # Initialize session state variables
    initialize_session_state()
    
    # Render navbar and navigate to selected page
    selected_page = render_navbar()

    if selected_page == "Process PDFs":
        process_pdfs_page()
    elif selected_page == "Process URLs":
        process_urls_page()
    elif selected_page == "Process YouTube Videos":
        process_youtube_page()


    # Add spacing using a divider
    st.divider()  # Adds a horizontal line and some spacing above it

    # Render the Start Over button at the bottom
    render_start_over_button()

if __name__ == "__main__":
    main()

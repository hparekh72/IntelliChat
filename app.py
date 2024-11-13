import streamlit as st

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        # Allow multiple PDF files to be uploaded
        uploaded_files = st.file_uploader("Upload your PDFs here and click on 'Process'", 
                                          type="pdf", 
                                          accept_multiple_files=True)
        if st.button("Process"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Process each PDF file
                    st.write(f"Processing file: {uploaded_file.name}")
                    # Here you can add code to read and process each PDF
            else:
                st.warning("Please upload at least one PDF file before clicking 'Process'.")

if __name__ == '__main__':
    main()

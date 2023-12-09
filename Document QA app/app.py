import streamlit as st
import shutil
import tempfile
import io
import os
import time
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import pinecone

# Set OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "ADD KEY"

# Embedding model
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

# Initialize Pinecone
pinecone.init(
    api_key="add key",
    environment="gcp-starter"
)
index_name = "out index name"

# Read the file
def read_doc(file_obj):
    if not isinstance(file_obj, io.BytesIO):
        raise TypeError("File object must be a BytesIO object")

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "uploaded_file.pdf")
    
    with open(file_path, "wb") as f:
        f.write(file_obj.getvalue())

    file_loader = PyPDFDirectoryLoader(tmp_dir)
    documents = file_loader.load()

    shutil.rmtree(tmp_dir)
    return documents

# Chunk data
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

# Retrieve query
def retrieve_query(query, k=3):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

# Load QA chain
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)
chain = load_qa_chain(llm, chain_type="stuff")

# Retrieve answers
def retrieve_answers(query):
    doc_search = retrieve_query(query)
    response = chain.run(input_documents=doc_search, question=query)
    return response

# Set page configuration
st.set_page_config(
    page_title="Document Reader and QA App",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main Streamlit app
def main():
    st.title("Document Reader and QA App")

    # Upload file
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    # Remove file button
    if uploaded_file and st.button("Remove File"):
        uploaded_file = None

    # If file is uploaded
    if uploaded_file:
        st.success("File successfully uploaded!")

        # Read and process the document
        docs = read_doc(uploaded_file)
        documents = chunk_data(docs=docs)

        # Create Pinecone index if documents are available
        global index
        with st.spinner("Creating Pinecone index..."):
            index = Pinecone.from_documents(documents, embeddings, index_name=index_name)

        # User input for querying
        our_query = st.text_input("Enter your query:")

        # Get answers button
        if st.button("Get Answers"):
            if index:
                # Display processing message
                with st.spinner("Retrieving answers..."):
                    # Simulate processing delay
                    time.sleep(2)

                    # Retrieve and display answers
                    answer = retrieve_answers(our_query)
                    st.subheader("Answers:")
                    st.write(answer)
            else:
                st.warning("Please upload a file before getting answers.")

if __name__ == "__main__":
    main()

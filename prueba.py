import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
    add_vertical_space(5)
    st.write('Modified with 💪 by [Manrrolo](https://manrrolo.github.io/Page/)')

load_dotenv()

# Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

def main():
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))
            service_context = ServiceContext.from_defaults(llm_predictor=modelo)
            index = GPTVectorStoreIndex.from_documents(VectorStore, service_context = service_context)
            
            respuesta = index.as_query_engine().query(query + " Responde en español")
            st.write(respuesta.response)

if __name__ == '__main__':
    main()

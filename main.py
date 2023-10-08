__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


button(username="statsholic", floating=True, width=211)

load_dotenv()

st.title("CHATPDF")
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


uploaded_file = None

if not uploaded_file:
    uploaded_file = st.file_uploader("Choose your PDF file.", type=["pdf"])
    st.write("---")

if uploaded_file:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30, length_function=len, is_separator_regex=False
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings_model)

    st.header("Ask about the PDF")
    with st.form(key="query_form"):
        question = st.text_input("Your question")

        # 'Ask' button
        submit_button = st.form_submit_button(label="Ask")

        # If the form is submitted (either by pressing enter or clicking 'Ask')
        if submit_button:
            with st.spinner("Wait for it..."):
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    streaming=True,
                    callbacks=[stream_handler],
                )
                qa_chain = RetrievalQA.from_chain_type(retriever=db.as_retriever(), llm=llm)
                answer = qa_chain({"query": question})
                # st.write(answer["result"])
                print(answer)

# relevent documents
# retriver_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
# relevant_documents = retriver_from_llm.get_relevant_documents(query=question, top_k=1)
# print(len(relevant_documents))
# print(relevant_documents)

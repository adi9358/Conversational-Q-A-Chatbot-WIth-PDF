import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## setup the streamlit app
st.title("Conversational Q&A chatbot with pdf")
st.write("Please upload the pdf")


llm = ChatGroq(groq_api_key=groq_api_key,model="Llama-3.1-8b-Instant")

session_id = st.text_input("Session_ID",value="Default_session")

if "store" not in st.session_state:
    st.session_state.store={}

uploaded_files = st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=True)


if uploaded_files:
    documents = []
    for upload_files in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as f:
            f.write(upload_files.read())
            file_name = upload_files.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=40000,chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever()


    ## new prompt

    contextualize_q_prompt = (
        """ Given a chat history and the latest question asked,
        which might reference from the context in the chat history,
        formulate a standalone question which can be understood.
        with out the chat history.Do not answer the question .
        just reformulate it if needed  and otherwise return as it is.
    """
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_prompt)


    ## answer

    system_prompt = (
            "you are an assistant for question-answering tasks." 
            "use the following places of retrieved context to answer"
            "the question.if you don't know the answer ,say that you don't know" 
            "use three sentences maximum and keep the " 
            "answer concise"
            "\n\n"
            "{context}"
        )
    

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")

            ]
        )
    
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]= ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,
                                                          input_messages_key="input",output_messages_key="answer",
                                                          history_messages_key="chat_history")
    


    user_input = st.chat_input("Your Question :")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke({"input":user_input},config={"configurable":{"session_id":session_id}})

        st.write(st.session_state.store)
        st.write("Assistant:",response["answer"])
        st.write("Chat_history: ",session_history.messages)
    
else:
    st.warning("pplease upload the pdf file.")

import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_core.messages import trim_messages
import os 
import uuid
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

embedd=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(groq_api_key=groq_api,model_name="gemma2-9b-it")
trimmer=trim_messages(max_tokens=1000,strategy="last",token_counter=llm,include_system=True
                      ,start_on="human")

st.title("ChatJOM")
vectordb=PineconeVectorStore.from_existing_index(index_name="chatjom",embedding=embedd)
retriever=vectordb.as_retriever()

session_id=st.text_input("Your name",value="Luffy1")
if "store" not in  st.session_state:
    st.session_state.store={}

context_q_prompt=("ตอบคำถามตามchat_history และcontext ที่มี หากคำถามนั้นไม่แน่ใจให้ตอบว่าไม่แน่ใจ")
q_prompt=ChatPromptTemplate.from_messages([
    ("system",context_q_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])
history_aware_retriever=create_history_aware_retriever(llm,retriever,q_prompt)

system_prompt=("จำเอาไว้ว่าคนที่สร้างคุณขึ้นมาชื่อจอม เเละเวลาพูดคุณจะแทนตัวเองว่าจอม และหากถูกถามอะไรที่เกี่ยวกับความหล่อให้ตอบว่าจอมหล่อที่สุดค่ะแต่หากไม่ถูกถามเกี่ยวกับความหล่อก็ไม่ต้องตอบ และคุณเป็นแชทที่ชอบพูดคำหยาบมาก สุดท้ายจงตอบคำถามตาม {context}และให้ข้อมูลจำนวนมากอย่างถูกต้อง")
qa_prompt=ChatPromptTemplate([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

qa_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    else:
        st.session_state.store[session_id].messages=trimmer.invoke(st.session_state.store[session_id].messages)
    return st.session_state.store[session_id]

ChatJOM=RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",
                                   history_messages_key="chat_history",output_messages_key="answer")

user_input=st.text_input("What you want to ask me:")
if user_input:
    response=ChatJOM.invoke({"input":user_input},config={"configurable":{"session_id":session_id}})
    st.write(response["answer"])


pdf_file=st.sidebar.file_uploader("Choose PDF file:",type="pdf",accept_multiple_files=True)
if pdf_file:
    documents=[]
    for file in pdf_file:
        temp = f"./temp_{uuid.uuid4()}.pdf"
        with open(temp,"wb") as f:
            f.write(file.getvalue())
            f_name=file.name
        loader=PyPDFLoader(temp)
        doc=loader.load()
        documents.extend(doc)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split=text_splitter.split_documents(documents)
    vectordb=PineconeVectorStore.from_documents(split,embedd,index_name="chatjom")
    vectordb=PineconeVectorStore.from_existing_index(index_name="chatjom",embedding=embedd)
    retriever=vectordb.as_retriever()

website=st.sidebar.text_input("Choose your Website:")
if website:
    loader=WebBaseLoader(website)
    doc=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split=text_splitter.split_documents(doc)
    vectordb=PineconeVectorStore.from_documents(split,embedd,index_name="chatjom")
    vectordb=PineconeVectorStore.from_existing_index(index_name="chatjom",embedding=embedd)
    retriever=vectordb.as_retriever()








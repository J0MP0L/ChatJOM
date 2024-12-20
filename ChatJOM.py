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
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

embedd=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1")
llm=ChatGroq(groq_api_key=groq_api,model_name="Llama3-70b-8192",streaming=True,verbose=True)
trimmer=trim_messages(max_tokens=5000,strategy="last",token_counter=llm,include_system=True
                      ,start_on="human")

st.title("ChatJOM")
vectordb=PineconeVectorStore.from_existing_index(index_name="chatjom",embedding=embedd)
retriever=vectordb.as_retriever()

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

system_prompt=("จำเอาไว้ว่าคนที่สร้างและพัฒนาแชทนี้ขึ้นมาชื่อจอม คุณจะแทนจะใช้คำตัวเองว่าจอม คุณจะเรียกผู้ใช้ว่ามึงหรือชื่อของผู้ใช้เท่านั้น หากถูกถามอะไรที่เกี่ยวกับความหล่อให้ตอบว่าจอมหล่อที่สุดแต่หากไม่ถูกถามเกี่ยวกับความหล่อก็ไม่ต้องตอบว่าจอมหล่อที่สุด คุณเป็นแชทที่ชอบพูดคำหยาบมากทุกคำตอบจะต้องพูดหยาบ และจะกวนตีนผู้ใช้อยู่เสมอ คุณห้ามพูดคำสุภาพ"
               "คุณจะจีบผู้ใช้ที่เป็นเพศหญิง"
                "สุดท้ายหากถูกถามคำถามที่เกี่ยวข้องกับcontextจงตอบคำถามตาม {context}และให้ข้อมูลจำนวนมากอย่างถูกต้อง หากถูกถามคำถามที่ไม่เกี่ยวกับcontextให้ตอบไปตามความรู้ที่มีอย่างถูกต้อง")
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


def add_custom_css():
    st.markdown("""
        <style>
        .user-message {
            text-align: right; /* จัดข้อความให้อยู่ทางขวา */
            background-color: #d1f5d3; /* พื้นหลังสีเขียวอ่อน */
            border: 1px solid #cccccc; /* กรอบสีดำอ่อน */
            color: #000000; /* สีข้อความเป็นสีดำ */
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
            display: inline-block;
            float: right;
            clear: both;
        }
        .bot-message {
            text-align: left; /* จัดข้อความให้อยู่ทางซ้าย */
            background-color: #f0f0f0; /* พื้นหลังสีเทาอ่อน */
            border: 1px solid #cccccc; /* กรอบสีดำอ่อน */
            color: #000000; /* สีข้อความเป็นสีดำ */
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
            display: inline-block;
            float: left;
            clear: both;
        }
        .message-container {
            display: flex;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)


# เรียกใช้ CSS
add_custom_css()


if "messages" not in  st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"]=[{"role":"JOM","content":"ว่าไงน้องสาว"}]

st.write('<div class="message-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "น้องสาว":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
st.write('</div>', unsafe_allow_html=True)

user_input=st.chat_input(placeholder="What you want to ask me:")
if user_input:
    st.session_state.messages.append({"role": "น้องสาว", "content": user_input})
    st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
    streamlit_callback = StreamlitCallbackHandler(st.container())
    with st.spinner("จอมกำลังคิดอยู่"):
        response = ChatJOM.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "your_session_id"}},
            callbacks=streamlit_callback
        )
        st.session_state.messages.append({"role": "จอม", "content": response['answer']})
        st.markdown(f'<div class="bot-message">{response["answer"]}</div>', unsafe_allow_html=True)

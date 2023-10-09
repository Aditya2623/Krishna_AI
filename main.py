import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle
from sentence_transformers import SentenceTransformer
from apikey import api
import os 
from langchain.llms import OpenAI


def getvectors(prompt):
    file_path="vector.pkl"
    if os. path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex= pickle.load(f)
    model =SentenceTransformer("all-mpnet-base-v2")
    search=prompt
    vec=model.encode(search)
    svec=np.array(vec).reshape(1,-1)   
    return vectorIndex.search(svec,k=2)
def get_text(prompt):
    vecindx=getvectors(prompt)
    with open ('gita.txt') as f:
        content=f.read()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
    )
    texts=text_splitter.split_text(content)
    return texts[vecindx[1][0][0]]+texts[vecindx[1][0][1]]
    
def get_LLMop(prompt):
    os.environ['OPENAI_API_KEY'] = api
    llm = OpenAI(temperature=0.6)
    ans=llm(get_text(prompt)+'using this texts tell me {prompt} respond as you are Lord Krishna and asuume i am as ur devotee/son/disciple.keep it short and highly creative')
    return ans

st.title("Krishna AI")

image = Image.open('assets/icon.png')
image2 = Image.open('assets/icon2.png')
#message = st.chat_message(name='Krishna',avatar=image)
with st.chat_message("Krishna",avatar=image):
        st.markdown("Hey Bhakth,how can I help you")

if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
prompt=st.chat_input('Ask me for moksha')
if prompt:
    with st.chat_message("user",avatar=image2):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    response=get_LLMop(prompt)
    with st.chat_message("Krishna",avatar=image):
        st.markdown(response)
    st.session_state.messages.append({"role":"Krishna","content":response})
import streamlit as st
import numpy as np
from PIL import Image
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
import pinecone
from apikey import llmkey, vec_db
import os
#from langchain.llms import OpenAI
#from langchain.llms import GooglePalm
import google.generativeai as genai


def get_text(prompt): 
    vec_db()
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    input_em = model.encode(prompt).tolist()
    index=pinecone.Index('krishnaai')
    output=index.query(input_em,top_k=2,includeMetadata=True).to_dict()  
    print(output['matches'][0]['metadata']['text']+output['matches'][1]['metadata']['text'])
    return output['matches'][0]['metadata']['text']+output['matches'][1]['metadata']['text']
    
def get_LLMop(prompt):
    os.environ["GEMINI_API_KEY"]=llmkey
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key = gemini_api_key)
    #llm=GooglePalm()

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Who is the GOAT in the Football?")

    ans=model.generate_content([get_text(prompt)+'use these text if possible and guide me {prompt}. Answer me as you are Lord Krishna,answer should be should be very short,highly creative and with a simple English'])
    return ans.text

st.title("Krishna AI")

image = Image.open('icon.png')
image2 = Image.open('icon2.png')
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
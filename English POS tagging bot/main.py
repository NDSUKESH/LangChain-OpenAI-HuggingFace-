## Required Library
from langchain import OpenAI,LLMChain
import streamlit as st
import os
from langchain.prompts import PromptTemplate

## setting openAi key
os.environ["OPENAI_API_KEY"]="HERE ADD YOUR OWN OPEN API KEY " ########
##Function for hitting OpenAPI and get response

## define the promt template required for the curent use chatbot
prompt_template=PromptTemplate(input_variables=['word'],
template="Tell me the part of speech for the given word {word} and give it a simple example")
def get_response(word):
    llm=OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model_name="text-davinci-003",temperature=0.6)
    chain=LLMChain(llm=llm,prompt=prompt_template)
    a=chain(word)
    return a

##webpage header
st.set_page_config(page_title="PartofSpeech chatbot")
st.header("Langchain PartOfSpeech Application")

##geting input 
input=st.text_input("Input:",key="input")

#passing to funtion
response=get_response(input)
submit=st.button("Type the Word to find the POS")

if submit:
    st.subheader("The response is")
    st.write(response)
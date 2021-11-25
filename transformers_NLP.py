#Build an application using streamlit that allows users to select the available pipelines in transformers and perform the NLP operations.
#import libraries (transformers and streamlit in this case)
from transformers import pipeline
from operator import itemgetter
import streamlit as st
#import numpy as np
#import pandas as pd
#setup streamlit app to take user input and offer the different transformers in the sidebar
template = """<div style = "background-color:red; padding:1px;">
                <h2 style = "color=:white; text-align:center">Transformers App </h2>
                </div>""" #allows multiple lines of html
#css colour codes available online, basic css
st.markdown(template,unsafe_allow_html=True) #tells streamlit to run the home written "unsafe" html above
st.write("")
st.subheader("This app will show you how computers can work with your inputs in different ways")
st.write("")
st.write("This app will help you analyse a statement using Natural Language Processing (a type of machine learning)")
st.write("")
st.write("")
st.write("*First of all, select the type of transformer you'd like to try out on the left hand side*")
st.write("*Then you will see the details of what you have to enter*")
st.write("")
st.sidebar.title("Select what you would like to try out")
dropdown = st.sidebar.selectbox("Select one",["","Sentiment Analysis","Text Generation","NER", "Question Answering","Summarization","Blank filling"])

#perform a different transformation depending on the option selected
if dropdown == "Sentiment Analysis":
    input_text_sent = st.text_input(label="Please enter your statement here so you can learn about the feeling of it")
    sentiments = pipeline("sentiment-analysis")
    sentiment = sentiments(input_text_sent)
    feeling = sentiment[0].get('label')
    st.write("the sentence you have entered is: ",feeling)
elif dropdown == "Text Generation":
    input_text_nlg = st.text_input(label="Please enter the sentence that you'd like to have an ending for")
    NLG = pipeline("text-generation")
    text_prompt = NLG(input_text_nlg)
    finished_sentence = text_prompt[0].get('generated_text')
    st.write("*Here is the suggested ending to your sentence:*")
    st.write(finished_sentence)
elif dropdown == "NER":
    input_text_ner = st.text_input(label="Please enter the sentence that you'd like to understand the parts of speech for")
    st.write("Hint: include the names of people, places and/or organisations for the best results")
    named_e_r = pipeline("ner")
    NER = named_e_r(input_text_ner)
    for i in range(len(NER)):
        st.write(str(NER[i]))
elif dropdown == "Question Answering":
    question = st.text_input(label="Please enter the question you'd like an answer for")
    context = st.text_input(label="Please provide some context for your question.  This should contain the answer - our computer isn't that clever yet!")
    Q = pipeline("question-answering")
    text_prompt_quest = Q(question,context)
    answer = text_prompt_quest.get('answer')
    st.write(answer)
elif dropdown == "Summarization":
    input_text_summ = st.text_area(label="Please enter the paragraph you'd like to have a summary for")
    summ = pipeline("summarization",min_length=3)
    text_prompt_summ = summ(input_text_summ)
    finished_summary = text_prompt_summ[0].get('summary_text')
    st.write("*This is the summary of your text:*")
    st.write(finished_summary)
elif dropdown == "Blank filling":
    input_text_mask = st.text_input(label="Please enter the sentence with one word replaced by <mask>")
    st.write("You must use <mask> specifically (with the < before and > after) in your text or this won't work")
    st.write("You will now see 2 possible words that could be used to fill in the blank or masked word")
    mask = pipeline("fill-mask", top_k=2)
    text_prompt_mask = mask(input_text_mask)
    suggestion_1 = text_prompt_mask[0].get('token_str')
    suggestion_2 = text_prompt_mask[1].get('token_str')
    st.write("The 2 suggested words for your blank are:")
    st.write("*Suggestion 1: *",suggestion_1)
    st.write("*Suggestion 2: *", suggestion_2)


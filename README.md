this is code run in colab for project: Text & Natural Language Processing (NLP) - Twitter Sentiment Analysis

file app-textbased is for text based output
file app-text is for output gui run in with ngrok server 

step to run app-text is 
1. import getpass
ngrok_key = getpass.getpass("Enter ngrok key: ")
2. !pip install streamlit pyngrok
3. %%writefile app.py
import streamlit as st ( copy file app-text into app.py)
4. from pyngrok import ngrok
port = 8501
ngrok.set_auth_token(ngrok_key)
ngrok.connect(port).public_url
5. #run stremline
!streamlit run app.py --server.port 8501 --server.enableCORS false




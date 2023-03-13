import streamlit as st
from readme_chat import print_answer

st.title('MongoDB IaC Chat Interface!')
# key = st.text_input('Open Api Key', type="password")
# os.environ["OPENAI_API_KEY"]= key
question = st.text_input('Ask your question')



if st.button("Submit") and question !="":
        # Call the OpenAI API to generate text
        response = print_answer(question)    
        st.write("Response:")
        st.write(response)
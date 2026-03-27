import openai
from langchain.llms import OpenAI
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
import io

def main():
    load_dotenv()

    st.set_page_config(page_title="Let's Start Analysing 💡")
    st.header("Let's Start Analysing 💡")

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Make a preliminary query on your CSV:")

        # Initialize the OpenAI model
        llm = OpenAI(temperature=0)
        agent = create_csv_agent(llm, user_csv, verbose=True,allow_dangerous_code=True)

        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()

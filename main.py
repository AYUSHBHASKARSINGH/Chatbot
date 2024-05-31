import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
# FAISS FOR SEMANTIC SEARCH AFTER MAKING VECTOR DATABASE TO MATCH 
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check for .txt files only
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                text += f.read()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    # print(chunks)
    # print('\n')
    # for i,_ in enumerate(chunks):
    #     print(f"{len(chunks[i])} -- {chunks[i]}")
    return chunks



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



    # in addition to answer provide a summary  based on what you think is crux of text
    # just reply that the answer is not available in text do not give false information
def get_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context answer the most suitable and relatable word sentence related to question
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # chain = load_qa_chain(model, chain_type="map-reduce", prompt=prompt)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(query)

    chain = get_chain()

    response = chain(
        {"input_documents": docs, "question": query}
        , return_only_outputs=True)

    return response
    print(response)
    # st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("Chat Text Files")
    st.header("Ask Me")

    folder_path = "Data" 

    # user_question = st.text_input("Ask a Question from the Text Files")

    # Setup a session state message variable to hold all the old messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # # Display all the historical messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    
    user_question = st. chat_input( 'Pass Your Prompt here')
    # If the user hits enter then
    if user_question:
        # Display the prompt
        st.chat_message('user').markdown(user_question)
        st.session_state.messages.append({'role':'user','content':user_question})
        with st.spinner("Processing..."):
            text = get_text(folder_path)
            text_chunks = get_chunks(text)
            get_vector_store(text_chunks)
            response = user_input(user_question)
            st.chat_message('assistant').markdown(response["output_text"])
            st.session_state.messages.append(
                {'role':'assistant','content':response["output_text"]}
            )



if __name__ == "__main__":
    main()
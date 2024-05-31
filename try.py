# from langchain. document_loaders import PyPDFLoader 
# from langchain.indexes import VectorstoreIndexCreator 
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings 
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # Bring in streamlit for UI dev


# import streamlit as st

# st.title('Ask watsonx')
# # Build a prompt input template to display the prompts


# # Setup a session state message variable to hold all the old messages
# if 'messages' not in st.session_state:
#     st.session_state.messages = []


# # Display all the historical messages
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])


# prompt = st. chat_input( 'Pass Your Prompt here')
# # If the user hits enter then
# if prompt:
#     # Display the prompt
#     st.chat_message('user').markdown(prompt)
#     st.session_state.messages.append({'role':'user','content':prompt})




# import streamlit as st
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# # from langchain.vectorstores import FAISS
# # FAISS FOR SEMANTIC SEARCH AFTER MAKING VECTOR DATABASE TO MATCH 
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_text_from_folder(folder_path):
#     text = ""
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):  # Check for .txt files only
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, "r") as f:
#                 text += f.read()
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# chunks1 = get_text_chunks(''' 
#     The year 2035 is anticipated to witness an unprecedented surge in internet demand, catalysed by rapid technological advancements and the proliferation of connected devices. This growth in internet consumption will have far-reaching implications for global communications and will reshape the way individuals, businesses, and societies interact with each other. In this article, we explore Internet Demand2035, its impact on communications, and the alternative technologies that will play crucial roles in shaping the digital landscape of the future.
# 1. The Exponential Rise in Internet Demand:
# By 2035, the global population will become increasingly reliant on the Internet for various aspects of daily life. The proliferation of smartphones, tablets, smart home devices, and IoT-enabled appliances will significantly contribute to the exponential rise in internet demand. Emerging economies, with their increasing access to affordable internet services, will become significant drivers of this trend, further accelerating the demand for online connectivity.
# 2. Evolution of Communications:
# The surge in internet demand will revolutionize traditional communication methods. Voice and video calling will be more seamless and high-quality due to the widespread implementation of 5G networks. Real-time communication will become more immersive and interactive, enabling virtual meetings and conferences to rival in-person interactions. This will result in increased productivity and reduced travel costs for businesses and individuals alike.
# 3. The Role of AI in Communications:
# By 2035, Artificial Intelligence (AI) will play a pivotal role in optimizing communication experiences. AI-driven chatbots and virtual assistants will become ubiquitous, streamlining customer support and enhancing the overall user experience. AI-powered language translation services will break down language barriers, facilitating cross-cultural communication and fostering global connections.
# ''')

# for i,_ in enumerate(chunks1):
#     print(f"chunk #{i}, size: {len(chunks1[i])}")


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just reply that the answer is not available in text do not give false information\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                                    temperature=0.3)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])


# def main():
#     st.set_page_config("Chat Text Files")
#     st.header("Chat with Text Files using Gemini")

#     folder_path = "Data" 

#     user_question = st.text_input("Ask a Question from the Text Files")

#     if user_question:
#         with st.spinner("Processing..."):
#             text = get_text_from_folder(folder_path)
#             text_chunks = get_text_chunks(text)
#             get_vector_store(text_chunks)
#             user_input(user_question)



# if __name__ == "__main__":
#     main()

































































import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check for .txt files only
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                text += f.read()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just reply that the answer is not available in text do not give false information\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, conversation_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
 
    response = chain(
        {"context": docs, "question": user_question}
        , return_only_outputs=True)

    conversation_history.append({"user_input": user_question, "answer": response["output_text"]})
    st.session_state["conversation_history"] = conversation_history

    print(response)
    return response["output_text"]


def main():
    st.set_page_config("Chat Text Files")
    st.title("Chat with Text Files using Gemini")

    folder_path = "Data"  # Replace with your actual folder path

    # Initialize conversation history from session state (if available)
    conversation_history = st.session_state.get("conversation_history", [])

    # Display conversation history
    if conversation_history:
        with st.expander("Conversation History"):
            for item in conversation_history:
                st.write(f"You: {item['user_input']}")
                st.write(f"Gemini: {item['answer']}")
                st.write("---")

    user_question = st.text_input("Ask a Question from the Text Files")

    if user_question:
        with st.spinner("Processing..."):
            text = get_text_from_folder(folder_path)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            answer = user_input(user_question, conversation_history.copy())
            st.write("Reply:", answer)


if __name__ == "__main__":
    main()









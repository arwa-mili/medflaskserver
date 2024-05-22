import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np


load_dotenv()

AZURE_ENDPOINT = os.environ['AZURE_ENDPOINT']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
DEPLOYEMENT_NAME = os.environ['DEPLOYEMENT_NAME']

app = Flask(__name__)


model = load_model('model_003.h5')



class chat_gen():
    def __init__(self):
        self.chat_history=[]


    def load_doc(self,document_path):
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        # Split document in chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # Create vectors
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Persist the vectors locally on disk
        vectorstore.save_local("faiss_index_datamodel")

        # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_datamodel", embeddings, allow_dangerous_deserialization=True)
        return persisted_vectorstore


    def load_model(self,):
        llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=DEPLOYEMENT_NAME,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=OPENAI_API_KEY,
            streaming=True,
        )

        system_instruction = """ As an AI assistant, you must answer the query from the user from the retrieved content, on each question and each part of discussion
        if no relavant information is available, if the topic is related to pdf , answer using your own knowledge , else say that the topic is not in the context!, also try not to invent things !!"""

        template = (
            f"{system_instruction} "
            "Combine the chat history{chat_history} and follow up question into "
            "a standalone question to answer from the {context}. "
            "Follow up question: {question}"
        )

        prompt = PromptTemplate.from_template(template)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.load_doc("./data/pat116_diabetes_18_large.pdf").as_retriever(),

            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )
        return chain

    def ask_pdf(self,query):
        result = self.load_model()({"question":query,"chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result['answer']




chat = chat_gen()



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Define the sample data
    sample_data = np.array([[
        data["Family_History_Diabetes"],
        data["Family_History_Cardiovascular_Disease"],
        data["Gender"],
        data["Heart_Rate_Change"],
        data["Systolic_Blood_Pressure"],
        data["Diastolic_Blood_Pressure"],
        data["HbA1c"],
        data["HDL"],
        data["TC_HDL_Ratio"]
    ]])

    # Assuming model is defined earlier in your Flask app
    prediction = model.predict(sample_data)
    return jsonify(prediction.tolist())




@app.route('/chat', methods=['POST'])
def chatbot():
    query = request.json['question']

    response = chat.ask_pdf(query)
    print(response)

    return jsonify({'answer': response})


if __name__ == '__main__':
    app.run(port=5000,debug=True)
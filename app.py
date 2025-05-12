from flask import Flask , render_template , jsonify , request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.prompt import *
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embedding = download_hugging_face_embedding()

index_name = "medical-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embedding,
)
retriever = docsearch.as_retriever(search_type = 'similarity' , search_kwargs={"k": 3})

llm = OpenAI(temperature = 0.4 , max_tokens = 500)


prompt  = ChatPromptTemplate.from_messages(
    [
        ('system' , system_prompt),
        ('human' , "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm , prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get' , methods = ['GET' , 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ", response['answer'])
    return str(response['answer'])


if __name__ == '__main__':
    app.run(host = "0.0.0.0" , port = 8080 , debug = True)




import os
import openai

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load documents from github repo
root_dir = './tf-aws-django-website'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass


# chunk the files
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

# create vector store
db = Chroma.from_documents(docs, embeddings)
db.add_documents(texts)

# create retriever
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# create conversational chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# model = ChatOpenAI(model_name='gpt-4') # switch to 'gpt-4'
model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=.25) # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

# ask questions
questions = [
        "What is this repository about?",
        "Describe the use of middleware in settings.py, what is important to know?",
        "Look for security issues in the code, settings.py or various configs. Provide details on them to me.",
        "How is it utilizing nginx?",
        "Explain the class based views for the django blog application.",
        "If I am new to django, what are the top 5 things I should know?",
        "Tell me a django joke."
        ]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")

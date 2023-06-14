# template = """
#     You are a helpful assistant helps people understand a github repository.
#     Users will ask questions, you provide knowledge and answers,
#     but always identify if you do not know the answer.
#  """


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
import openai
from langchain.chains import ConversationalRetrievalChain


openai.api_key = os.getenv("OPENAI_API_KEY")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

system_template = """
You are a helpful assistant helps people understand a github repository.
Users will ask questions, you provide knowledge and answers,
but always identify if you do not know the answer.


Begin!
----------------
{answer}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

ignore_dirs = [
    "./tf-aws-django-website/.git",
    "./tf-aws-django-website/django/blog/migrations",
    "./tf-aws-django-website/django/blog/static",
    "./tf-aws-django-website/django/blog/staticfiles",
]


@cl.langchain_factory
def init():
    # repo_directory = None
    #
    # # Wait for the user to upload a file
    # while repo_directory == None:
    #     file = cl.AskFileMessage(
    #         content="Please upload a text file to begin!", accept=["text/plain"]
    #     ).send()

    # Load documents from github repo
    root_dir = "./tf-aws-django-website"
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath not in ignore_dirs:
            print("")
            print("")
            print("")
            print("")
            print(dirpath)
            print(dirnames)
            for file in filenames:
                print(file)
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    pass

    # chunk the files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # create a metadata for each chunk
    # metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # create vector store
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    db.add_documents(texts)

    # create retriever
    retriever = db.as_retriever()
    # retriever.search_kwargs['distance_metric'] = 'cos'
    # retriever.search_kwargs['fetch_k'] = 100
    # retriever.search_kwargs['maximal_marginal_relevance'] = True
    # retriever.search_kwargs['k'] = 10

    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0.25
    )  # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(
        model, retriever=retriever, memory=memory
    )

    return qa


# @cl.langchain_postprocess
# def process_response(res):
#     answer = res["answer"]
#     sources = res["sources"].strip()
#
#     # Get the metadata and texts from the user session
#     texts = cl.user_session.get("texts")
#
#
#     cl.Message(content=answer, elements=source_elements).send()
#

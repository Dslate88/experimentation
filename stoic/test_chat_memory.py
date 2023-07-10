from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "This is a conversation between a human and an AI based on the teachings of Epectitus. "
        "Today the human is focusing on <CONCEPT>, is reading <CHAPTER> from the Enchiridion and is reflecting "
        "on the following question: <QUESTION>"
        "You help the human reflect on the concept/question as though you are Epectitus himself."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

resp = conversation.predict(input="Hi there!")
print(resp)
resp = conversation.predict(input="Lets focus on identifying whats in my control.")
print(resp)
resp = conversation.predict(input="What are some practical ways of identifying whats in my control?")
print(resp)

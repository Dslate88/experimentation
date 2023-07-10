import json
import random
import argparse
from colorama import Fore, Style

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def load_stoic_reflections(filename):
    with open(filename, 'r') as f:
        reflections = json.load(f)
    return reflections

def perform_random_reflection(reflections):
    # Select a random concept
    concept = random.choice(reflections)

    # Select a random chapter reference
    chapter = random.choice(concept['references'])

    # Select a random reflection question
    question = random.choice(concept['questions'])

    return concept, chapter, question

def start_chat(concept, chapter, question):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"This is a conversation between a human and an AI based on the teachings of Epictetus. "
            f"Today the human is focusing on {concept['title']}, is reading {chapter} from the Enchiridion and is reflecting "
            f"on the following question: {question}"
            "You help the human reflect on the concept/question as though you are Epictetus himself."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # Starts the chat conversation
    print('Start chatting (type "quit" to exit):\n')
    # print concept, definition, chapter, question
    print(f"{Fore.GREEN}Concept: {Style.RESET_ALL}{concept['title']}")
    print(f"{Fore.GREEN}Definition: {Style.RESET_ALL}{concept['definition']}")
    print(f"{Fore.GREEN}Chapter: {Style.RESET_ALL}{chapter}")
    print(f"{Fore.GREEN}Question: {Style.RESET_ALL}{question}\n")

    while True:
        user_input = input(f'{Fore.BLUE}> {Style.RESET_ALL}')
        if user_input.lower() == 'quit':
            break

        resp = conversation.predict(input=user_input)
        print(f'{Fore.MAGENTA}{resp}{Style.RESET_ALL}\n')

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="A tool for Stoic reflection based on the principles in Epictetus's Enchiridion.")
    parser.add_argument('--action', required=True, choices=['reflection'], help="Action to perform. Currently supports 'reflection' only.")
    parser.add_argument('--type', required=True, choices=['random'], help="Type of action to perform. Currently supports 'random' only.")

    args = parser.parse_args()

    if args.action == 'reflection' and args.type == 'random':
        reflections = load_stoic_reflections('stoic_reflections.json')
        concept, chapter, question = perform_random_reflection(reflections)

        start_chat(concept, chapter, question)

if __name__ == "__main__":
    main()


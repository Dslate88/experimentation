import json
import random
import argparse

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

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="A tool for Stoic reflection based on the principles in Epictetus's Enchiridion.")
    parser.add_argument('--action', required=True, choices=['reflection'], help="Action to perform. Currently supports 'reflection' only.")
    parser.add_argument('--type', required=True, choices=['random'], help="Type of action to perform. Currently supports 'random' only.")

    args = parser.parse_args()

    if args.action == 'reflection' and args.type == 'random':
        reflections = load_stoic_reflections('stoic_reflections.json')
        concept, chapter, question = perform_random_reflection(reflections)

        print(f'Title: {concept["title"]}\nDefinition: {concept["definition"]}\nChapter to Read: {chapter}\nQuestion: {question}\n')

if __name__ == "__main__":
    main()


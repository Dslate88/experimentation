import openai
import chainlit as cl
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# model_name = "gpt-4"
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": 0.3,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# this doesnt make sense, but here for an example
prompt = """SQL tables (and columns):
* Customers(customer_id, signup_date)
* Streaming(customer_id, video_id, watch_date, watch_minutes)

A well-written SQL query that {input}:
```"""

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
def main(message: str):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message})
    print(message_history)

    fromatted_prompt = prompt.format(input=message)

    msg = cl.Message(content="",
                     llm_settings=cl.LLMSettings(model_name=model_name, **settings),
                     prompt=fromatted_prompt,
                     indent=2,
                     )

    response = openai.ChatCompletion.create(
        model=model_name, messages=message_history, stream=True, **settings
    )
    for resp in response:
        token = resp.choices[0]["delta"].get("content", "")
        msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    msg.send()

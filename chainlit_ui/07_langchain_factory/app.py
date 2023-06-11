from langchain import OpenAI, LLMMathChain, LLMBashChain
from chainlit import langchain_factory


@langchain_factory
def load():
    """
    Plug and play decorator for the LangChain library.
    The decorated function should instantiate a new LangChain instance (Chain, Agent…).
    One instance per user session is created and cached.
    The per user instance is called every time a new message is received.
    """
    llm = OpenAI(temperature=0)
    llm_math = LLMMathChain.from_llm(llm=llm)

    return llm_math

# NOTE: only one langchain_factory can be used per file
# @langchain_factory
# def test_load():
#     """
#     Plug and play decorator for the LangChain library.
#     The decorated function should instantiate a new LangChain instance (Chain, Agent…).
#     One instance per user session is created and cached.
#     The per user instance is called every time a new message is received.
#     """
#     llm = OpenAI(temperature=0.3)
#     llm_bash = LLMBashChain.from_llm(llm=llm)
#
#     return llm_bash

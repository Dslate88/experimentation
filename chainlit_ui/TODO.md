# features to explore

--------------------------------------------------------
# GRAND IDEA?
- approach: local instal/use for early stages
    - front-end where user logs in, once authenticated can see prior UI sessions
    - can select prior session to continue, or create a new session
    - this way the deployment of chainlit is simple, eh?

--------------------------------------------------------
## Value demonstration ideas for this weekend?

# vector db text generation (chunk text into vectors, put in vectorDB, ...GITHUB USECASE!!!)
- point at a repo, use langchain to ask a sheet ton of questions over it
  - https://python.langchain.com/en/latest/use_cases/code/twitter-the-algorithm-analysis-deeplake.html
  - point this at my terraform-aws-django-website repo??? use as my first post???
  - is it possible to link this to chainlit? for some flashy images in the post??

# sql chain example (answer questions over a database)
- https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html
- basically automate it all...point to the db with this functionality and it can create SQL

## phase 1: current thoughts on priority and approach
- start with vector db text generation pointing at my django website repo
- once I get that working, try to wrap chainlit over it
- put it on k8 on macbook??? extra shiny win for showing Mike + team

## phase 2: EKS use case optimization
- https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb
- following the format above, convert to EKS specific pattern, one that would benefit our team
- once thats working, connect it to a chainlit front-end

## phase 3: experimentation philosophy
- write down my persepctives on how to nail it then scale it

--------------------------------------------------------
# moderation
- https://python.langchain.com/en/latest/modules/chains/examples/moderation.html
- default functionality with customModeration too (ssn, stuff, and ...)

# self-critique chain with constitutional AI
- custom principles (do not talk about buying competitors products, etc)
- https://python.langchain.com/en/latest/modules/chains/examples/constitutional_chain.html

# router chains!
- https://python.langchain.com/en/latest/modules/chains/generic/router.html
- have a math totor chain, have a pyhsics tutor chain, ..., use LLM to route to the right one!!!!!

# hypotetical document embeddings
- https://python.langchain.com/en/latest/modules/chains/index_examples/hyde.html

# chat over documents with chat history
- https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html

# agent simulations
- https://python.langchain.com/en/latest/use_cases/agent_simulations.html
- value: have chatbot simulate converation about insurance claims

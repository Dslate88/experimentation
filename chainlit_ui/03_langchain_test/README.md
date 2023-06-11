# notes

## when using langchain with chainlit
- prompts and completions are cached locally! The cache lives in the .chainlit folder.
- the .chainlit dir is the root where the application is invoked, aka where app.py is located...
- also, each session overwrites this file..so it the user logs out, or creates a new chat its gone

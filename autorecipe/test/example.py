from autorecipe import AssistantAgent
from autorecipe import UserProxyAgent

proxAgt = UserProxyAgent(name="Agent")
LLMAgt = AssistantAgent(name="Chatbot")

proxAgt.initiate_chat(
    LLMAgt,
    message="""What date is today? Which big tech stock has the largest year-to-date return""",
)


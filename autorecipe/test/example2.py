import os

from genai.credentials import Credentials
from genai.extensions.huggingface.agent import IBMGenAIAgent
from genai.schemas import GenerateParams

api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = 'https://bam-api.res.ibm.com'

creds = Credentials(api_key, api_endpoint)
params = GenerateParams(min_new_tokens=10, max_new_tokens=200)

agent = IBMGenAIAgent(credentials=creds, model="meta-llama/llama-2-70b-chat", params=params)

#agent.chat("Download the text from the given url", url="https://research.ibm.com/blog/analog-ai-chip-low-power")
#agent.chat("Summarize the downloaded text")

agent.run("Use internet to find out different sensor variables associated with Wind turbine gearbox failure", return_code=False)
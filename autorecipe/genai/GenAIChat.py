from genai.credentials import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions
from genai.model import Model
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import Callable, Dict, List, Optional, Tuple, Union
import re
import json

UNKNOWN = "unknown"
def content_str(content: Union[str, List]) -> str:
    if type(content) is str:
        return content
    rst = ""
    for item in content:
        if item["type"] == "text":
            rst += item["text"]
        else:
            assert isinstance(item, dict) and item["type"] == "image_url", "Wrong content format."
            rst += "<image>"
    return rst


CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
def extract_code(
    text: Union[str, List], pattern: str = CODE_BLOCK_PATTERN, detect_single_line_code: bool = False
) -> List[Tuple[str, str]]:
    """Extract code from a text.

    Args:
        text (str or List): The content to extract code from. The content can be
            a string or a list, as returned by standard GPT or multimodal GPT.
        pattern (str, optional): The regular expression pattern for finding the
            code block. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable the new feature for
            extracting single line code. Defaults to False.

    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is code block but the language is not specified, the language would be "".
    """
    text = content_str(text)
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [(UNKNOWN, text)]

    # Extract both multi-line and single-line code block, separated by the | operator
    # `([^`]+)`: Matches inline code.
    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    # Extract the individual code blocks and languages from the matched groups
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    print (extracted)
    return extracted

class GenAIChatClient(Model):
    def __init__(self, model, params, credentials, system_message):
        self.client = LangChainChatInterface(
            model=model,
            params=GenerateParams(**params),
            credentials=Credentials(**credentials),
        )
        self._conversation_id = None
        self.system_message = system_message

    def _preprocess_create_payload(self, messages):
        chatmessage = []
        if self._conversation_id is None:
            chatmessage.append(SystemMessage(content=self.system_message))
        for item in messages:
            if item['role'] == 'system':
                chatmessage.append(SystemMessage(content=item['content']))
            elif item['role'] == 'user':
                chatmessage.append(HumanMessage(content=item['content']))
            elif item['role'] == 'assistant':
                chatmessage.append(AIMessage(content=item['content']))
        return [chatmessage]

    def create(self, context, messages):
        """ """
        messages = self._preprocess_create_payload(messages)
        if self._conversation_id:
            result = self.client.generate(
                messages=messages,
                options=ChatOptions(
                    conversation_id=self._conversation_id,
                    use_conversation_parameters=True,
                ),
            )
            return result.generations[0][0].text
        else:
            result = self.client.generate(messages=messages)
            self._conversation_id = result.generations[0][0].generation_info["meta"]["conversation_id"]
            return result.generations[0][0].text
        

    def extract_questions(self, text):
        chat_agent_response = content_str(text)
        questions_start_index = chat_agent_response.find("1. ")
        questions_end_index = chat_agent_response.rfind("\n\n") + 2  # Adding 2 to include the last newline characters
        questions_string = chat_agent_response[questions_start_index:questions_end_index]

        # Splitting the questions into a list
        questions_list = questions_string.split("\n")

        # Removing empty elements from the list
        questions_list = [question.strip() for question in questions_list if question.strip()]
        final_questions  = []
        for item in questions_list:
            first_space_index = item.find(' ')
            if first_space_index != -1:
                final_questions.append(item[first_space_index+1:])
            else:
                final_questions.append(item)

        # Printing the list of questions
        return final_questions


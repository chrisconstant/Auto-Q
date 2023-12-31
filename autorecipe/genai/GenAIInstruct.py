from genai.credentials import Credentials
from genai.schemas import GenerateParams
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions
from genai.model import Model
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import Callable, Dict, List, Optional, Tuple, Union
import re
import json
import mlflow
from genai.extensions.langchain import LangChainInterface
import time
import socket
from genai.exceptions.genai_exception import GenAiException

UNKNOWN = "unknown"


class GenAIInstructClient(Model):
    def __init__(
        self,
        name,
        description,
        skill,
        model,
        params,
        credentials,
        system_message,
        question_message,
    ):
        self.name = name
        self.description = description
        self.skill = skill
        self.client = LangChainInterface(
            model=model,
            params=GenerateParams(**params),
            credentials=Credentials(**credentials),
        )
        self.system_message = system_message
        self.question_message = question_message

        # track the request in and request out
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

        # trial
        self._max_retries = 3
        self._retry_delay = 10

    def _update_tokens_usage(
        self, prompt_tokens=0, completion_tokens=0, total_tokens=0
    ):
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += total_tokens

    def create(self, context, messages, experiment_id):
        """ """
        time.sleep(5)  # putting sleep for 5 second
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as conv:
            q_dict = {"Question": messages}
            mlflow.log_dict(q_dict, "Question.json")
            result = None
            for _ in range(1, self._max_retries + 1):
                try:
                    result = self.client.generate(
                        prompts=[
                            f"System Prompt: {self.system_message} \n\n {self.question_message} \n\n Question: {messages} Please use (Internal thought)."
                        ]
                    )
                    break
                except (OSError, socket.error, ConnectionResetError, Exception, GenAiException) as e:
                    print ('Error ....' + str(e))
                    time.sleep(self._retry_delay)

            if result:
                a_dict = {"Answer": result.generations[0][0].text}
                t_dict = result.generations[0][0].generation_info["token_usage"]
                self._update_tokens_usage(
                    t_dict["prompt_tokens"],
                    t_dict["completion_tokens"],
                    t_dict["total_tokens"],
                )
                mlflow.log_dict(a_dict, "Answer.json")
                return result.generations[0][0].text
            else:
                return ''

    def print_token_usage(self):
        print(
            f"The usages Promt Token: {self._prompt_tokens}, \
              Generated Token: {self._completion_tokens}, Total Token : {self._total_tokens}"
        )

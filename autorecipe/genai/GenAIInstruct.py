from genai.credentials import Credentials
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
import mlflow
import socket
import time
from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
import time
import socket
from genai import Client, Credentials

UNKNOWN = "unknown"


class GenAIInstructClient():
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
        stream=False,
    ):
        self.name = name
        self.description = description
        self.skill = skill
        ignored_key = 'moderations'
        filtered_params = {key: value for key, value in params.items() if key != ignored_key}

        self.llm = LangChainInterface(
            client=Client(credentials=Credentials(**credentials)),
            model_id=model,
            parameters=TextGenerationParameters(**filtered_params),
            moderations=params['moderations']
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

        # stream
        self.stream = stream

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
            chunk = None
            for _ in range(1, self._max_retries + 1):
                try:
                    if self.stream:
                        result = ''
                        for chunk in self.llm.stream(
                            input=[
                                f"System Prompt: {self.system_message} \n\n {self.question_message} \n\n Question: {messages} Please use (Internal thought)."
                            ]
                        ):
                            if result:
                                result = result + chunk
                            else:
                                result = chunk
                        
                    else:
                        result = self.llm.generate(
                            prompts=[
                                f"System Prompt: {self.system_message} \n\n {self.question_message} \n\n Question: {messages} Please use (Internal thought)."
                            ]
                        )
                    break
                except (OSError, socket.error, ConnectionResetError, Exception) as e:
                    print ('Error ....' + str(e))
                    time.sleep(self._retry_delay)

            if self.stream:
                if result:
                    a_dict = {"Answer": result.content.strip()}
                    t_dict = chunk.generation_info['token_usage']
                    self._update_tokens_usage(
                        t_dict["prompt_tokens"],
                        t_dict["completion_tokens"],
                        t_dict["total_tokens"],
                    )
                    return result.content.strip()
                else:
                    return ''
            else:
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

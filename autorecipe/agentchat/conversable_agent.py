from .agent import Agent
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from collections import defaultdict
import copy
from autorecipe.genai.GenAIChat import GenAIChatClient
from genai.model import Model

class ConversableAgent(Agent):
    MAX_CONSECUTIVE_AUTO_REPLY = 100
    DEFAULT_CONFIG = {
        "model": "meta-llama/llama-2-70b-chat",
        "params": {
            "decoding_method": "greedy",
            "min_new_tokens": 200,
            "max_new_tokens": 2000,
        },
        "creds": {
            "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
            "api_endpoint": "https://bam-api.res.ibm.com",
        },
    }

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        genai_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
    ):
        super().__init__(name)
        self._genai_messages = defaultdict(list)
        self._genai_system_message = [{"content": system_message, "role": "system"}]
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: x.get("content") == "TERMINATE")
        )

        if genai_config is False:
            self.genai_config = False
            self.client = None
        else:
            self.genai_config = self.DEFAULT_CONFIG.copy()
            if isinstance(genai_config, dict):
                self.genai_config.update(genai_config)
            self.client = GenAIChatClient(
                self.genai_config['model'],
                self.genai_config['params'],
                self.genai_config['creds'],
            )

        self._code_execution_config: Union[Dict, Literal[False]] = (
            {} if code_execution_config is None else code_execution_config
        )

        self.human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply
            if max_consecutive_auto_reply is not None
            else self.MAX_CONSECUTIVE_AUTO_REPLY
        )
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._max_consecutive_auto_reply_dict = defaultdict(
            self.max_consecutive_auto_reply
        )
        self._function_map = {} if function_map is None else function_map
        self._default_auto_reply = default_auto_reply
        self._reply_func_list = []
        self.reply_at_receive = defaultdict(bool)

        self.register_reply([Agent, None], ConversableAgent.generate_genai_reply)
        self.register_reply(
            [Agent, None], ConversableAgent.check_termination_and_human_reply
        )

    def register_reply(
        self,
        trigger: Union[Type[Agent], str, Agent, Callable[[Agent], bool], List],
        reply_func: Callable,
        position: int = 0,
        config: Optional[Any] = None,
        reset_config: Optional[Callable] = None,
    ):
        if not isinstance(trigger, (type, str, Agent, Callable, list)):
            raise ValueError(
                "trigger must be a class, a string, an agent, a callable or a list."
            )
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
            },
        )

    @property
    def system_message(self):
        return self._genai_system_message[0]["content"]

    def update_system_message(self, system_message: str):
        self._genai_system_message[0]["content"] = system_message

    def update_max_consecutive_auto_reply(
        self, value: int, sender: Optional[Agent] = None
    ):
        if sender is None:
            self._max_consecutive_auto_reply = value
            for k in self._max_consecutive_auto_reply_dict:
                self._max_consecutive_auto_reply_dict[k] = value
        else:
            self._max_consecutive_auto_reply_dict[sender] = value

    def max_consecutive_auto_reply(self, sender: Optional[Agent] = None) -> int:
        return (
            self._max_consecutive_auto_reply
            if sender is None
            else self._max_consecutive_auto_reply_dict[sender]
        )

    @property
    def chat_messages(self) -> Dict[Agent, List[Dict]]:
        return self._genai_messages

    def last_message(self, agent: Optional[Agent] = None) -> Optional[Dict]:
        if agent is None:
            n_conversations = len(self._genai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._genai_messages.values():
                    return conversation[-1]
            raise ValueError(
                "More than one conversation is found. Please specify the sender to get the last message."
            )
        if agent not in self._genai_messages.keys():
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._genai_messages[agent][-1]

    @staticmethod
    def _message_to_dict(message: Union[Dict, str]):
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    def _append_genai_messages(
        self, message: Union[Dict, str], role, conversation_id: Agent
    ) -> bool:
        message = self._message_to_dict(message)
        genai_message = {
            k: message[k]
            for k in ("content", "function_call", "name", "context")
            if k in message
        }
        if "content" not in genai_message:
            if "function_call" in genai_message:
                genai_message[
                    "content"
                ] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        genai_message["role"] = (
            "function" if message.get("role") == "function" else role
        )
        if "function_call" in genai_message:
            genai_message[
                "role"
            ] = "assistant"  # only messages with role 'assistant' can have a function call.
            genai_message["function_call"] = dict(genai_message["function_call"])
        self._genai_messages[conversation_id].append(genai_message)
        return True

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        print ('I am in send - it will be send to ' + recipient.name)
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_genai_messages(message, "assistant", recipient)
        print (valid)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def _process_received_message(self, message, sender, silent):
        message = self._message_to_dict(message)
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_genai_messages(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            pass
            # self._print_received_message(message, sender)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        print ('I am in recive of ' + self.name +  ' and reply will be send to ' + sender.name)
        self._process_received_message(message, sender, silent)
        if (
            request_reply is False
            or request_reply is None
            and self.reply_at_receive[sender] is False
        ):
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            print ('I am about to send response ' + reply)
            self.send(reply, sender, silent=silent)

    def _prepare_chat(self, recipient, clear_history):
        """This interaction clear all the info. 

        :param recipient: _description_
        :type recipient: _type_
        :param clear_history: _description_
        :type clear_history: _type_
        """
        self.reset_consecutive_auto_reply_counter(recipient)
        recipient.reset_consecutive_auto_reply_counter(self)
        self.reply_at_receive[recipient] = recipient.reply_at_receive[self] = True
        if clear_history:
            self.clear_history(recipient)
            recipient.clear_history(self)

    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ):
        """_summary_

        :param recipient: _description_
        :type recipient: ConversableAgent
        :param clear_history: _description_, defaults to True
        :type clear_history: Optional[bool], optional
        :param silent: _description_, defaults to False
        :type silent: Optional[bool], optional
        """
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def reset(self):
        """Reset the agent."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: Optional[Agent] = None):
        """Reset the reply_at_receive of the sender."""
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[sender] = False

    def reset_consecutive_auto_reply_counter(self, sender: Optional[Agent] = None):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if sender is None:
            print ('none - sender')
            self._consecutive_auto_reply_counter.clear()
        else:
            print ('----sender ', sender.name)
            self._consecutive_auto_reply_counter[sender] = 0

    def clear_history(self, agent: Optional[Agent] = None):
        if agent is None:
            self._genai_messages.clear()
        else:
            self._genai_messages[agent].clear()

    def generate_genai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Model] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._genai_messages[sender]

        # TODO: #1143 handle token limit exceeded error
        # print (self._genai_system_message + messages, messages[-1].pop("context", None))
        response = client.create(
            context=messages[-1].pop("context", None),
            messages=self._genai_system_message + messages,
        )
        # print (response)
        return True, response

    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Check if the conversation should be terminated, and if human reply is provided."""
        if config is None:
            config = self
        if messages is None:
            messages = self._genai_messages[sender]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = self.get_human_input(
                f"Provide feedback to {sender.name}. Last response was {message}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if (
                self._consecutive_auto_reply_counter[sender]
                >= self._max_consecutive_auto_reply_dict[sender]
            ):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender.name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        # if no_human_input_msg:
        #    print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        # if self.human_input_mode != "NEVER":
        #    print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    def generate_init_message(self, **context) -> Union[str, Dict]:
        """Generate the initial message for the agent.

        Override this function to customize the initial message based on user's request.
        If not overridden, "message" needs to be provided in the context.
        """
        return context["message"]

    def register_function(self, function_map: Dict[str, Callable]):
        """Register functions to the agent.

        Args:
            function_map: a dictionary mapping function names to functions.
        """
        self._function_map.update(function_map)

    def can_execute_function(self, name: str) -> bool:
        """Whether the agent can execute the function."""
        return name in self._function_map

    @property
    def function_map(self) -> Dict[str, Callable]:
        """Return the function map."""
        return self._function_map

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None]:
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._genai_messages[sender]

        for reply_func_tuple in self._reply_func_list:
            print (reply_func_tuple)
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply
    
    def _match_trigger(self, trigger, sender):
        """Check if the sender matches the trigger."""
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, Agent):
            return trigger == sender
        elif isinstance(trigger, Callable):
            return trigger(sender)
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")
        
    def get_human_input(self, prompt: str) -> str:
        """Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        reply = input(prompt)
        return reply
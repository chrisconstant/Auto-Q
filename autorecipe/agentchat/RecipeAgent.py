from autorecipe.genai.GenAIChat import GenAIChatClient
from autorecipe.genai.GenAIInstruct import GenAIInstructClient
from collections import defaultdict
import mlflow
import random
from autorecipe.genai.utils import (
    filter_and_sort_questions,
    filter_and_sort_questions_using_reference,
)
import pandas as pd
from colorama import Fore, Style
import uuid

# have model specific configuration
# QA does not need longer context to generate
# Max token generation need to be adjusted per

class RecipeAgent:
    # configuration
    DEFAULT_CONFIG = {
        #"model": "meta-llama/llama-2-70b-chat",
        "model": "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
        #"model": "ibm/granite-13b-chat-v2",
        "params": {
            "decoding_method": "greedy",
            "min_new_tokens": 200,
            "max_new_tokens": 2000, #1500,
            "stop_sequences": ["(TOKENSTOP)"],
        },
        "creds": {
            "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
            "api_endpoint": "https://bam-api.res.ibm.com",
        },
    }

    # update
    DSSystemPrompt = """
You are a helpful AI assistant. You act as a data scientist. Solve tasks using your coding and language skills. 
Your job is to build an anomaly model using real time time series sensor data
 obtained from IoT/OT system. In order to get domain understanding of the problem you will prepare a series of 
 questions to be asked in sequential orders to subject matter experts. Typical questions should focus on the
important components for which anomaly model should be build, the important failure modes and the ability of
 sensor data to detect these failure. You should also leverage the additional information made available in user message if any. 
 Please do not use a conversational approach to ask questions and gather information.
"""

    # Note: Please do not use a conversational approach to ask questions and gather information.
    # I'll ask follow-up questions based on the response I receive to ensure that I have a clear
    # understanding of the problem and the data.

    SMESystemPrompt = """
You act as a reliability engineer who is expert in failure modes and effect analysis (FMEA) of asset 
reliability. You task is to provide a accurate information about asset's component, subcomponent, failure mode
failure reason, failure code and degradation mechanisum along with severity and ability to detect them 
before it happen via preventive maintainance. You will be provided an enough information
about the asset class.
"""

    InfoSummaryPromt = """
Prepare a human-readable summary in a well-structured paragraph, eliminating any 
 special characters such as new lines and tabs. Focus on capturing the main components, failure modes and 
 reasons, degradation mechanisms, severity and detectability ratings, emphasizing the crucial insights for 
 predicting and preventing failures. Ensure the summary provides a coherent narrative. If user gives list of questions, 
 then summary should be written based on questions content for a given asset class. 
"""
# If user gives list of questions, then summary should be written based on questions content.
# Generate one paragraph summary. 
# Generate the background document to answer the given question.
    
    QuestionGenerator = """
Pretend you are a question generation system. I will give you a list of questions or a pair of question and answer 
extracted from the conversation between two users where question is asked by data scientist and 
subject matter expert has provided corresponding answer. Based on the conversation, you reply me with additional set of 
questions that data scientist can ask to subject matter expert. The newly generated questions must align with original set of questions. 
you should avoid generating duplicate questions. you should also avoid questions for which potential answer can be similar.
 Please do not use a conversational approach to ask questions and gather information.
"""

    QuestionClassifier = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
persona such as data scientists, subject matter experts, narrators etc. User will provide a 
question and you will select a persona who can answer the given question. Your selection is based 
on the persona's field experience and scientific knowledge. Sometime questions can be answered by 
multiple personas. Here is the two personas along with their skill description.  

Persona: Data Scientist
Skill: building machine learning model, data analytics, python programming

Persona: Subject Matter Expert
Skill: Provide domain knowledge for a particular industrial assets and their working condition
"""

    QuestionClassifierPropmt = """
Assuming user is seeking information about who can answer the following question. There is a possibility
that more than one persona can provide different level of information. Please generate only personas.

Question: How do you ensure that the wind turbine gearbox is properly aligned and balanced? What are the consequences of misalignment or imbalance, and how do you correct these issues?

Question: Can you discuss the role of condition monitoring in predicting and preventing wind turbine gearbox failures? What are the different types of condition monitoring techniques, and how do they help identify potential failures?

Question: Can you provide examples of common mistakes or oversights that can lead to wind turbine gearbox failures? How can these mistakes be avoided, and what are the consequences of not addressing them?

Question: Are there any additional data sources or information that can be leveraged to improve the accuracy of the anomaly model, such as historical data or expert knowledge? This will help me identify potential sources of additional information that can be used to improve the model.
"""

    AssetDescriptionExtractor = """
You are a helpful, respectful, and honest assistant. You will be provided a one line description that 
include industrial asset name and may include some configuration such as component name or configuration. You need  
to identify the device name which represent an industrial asset from the given description and then 
generate a device type and short device description. Provide answer in python json string with following keys : 
asset_class, asset_category, device_name, device_type and device_description. 

If you don't know the answer to a question, please don't share false information. Your answers should not 
include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that 
your responses are socially unbiased and positive in nature.
"""

    QAClassifierSystemPrompt = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
persona such as data scientists, subject matter experts, narrators etc. User will provide a 
question and you will select a persona who can answer the given question. Your selection is based 
on the persona's field experience and scientific knowledge. Sometime questions can be answered by 
multiple personas. Here is the two personas along with their skill description.  

Persona: Subject Matter Expert
Skill: Provide domain knowledge for a particular industrial assets and their working condition

Persona: Data Scientist
Skill: building machine learning model, data analytics, python programming

"""

    QAClassifierPrompt = """
Assuming user is seeking guidance about who can answer the given question. There is a possibility
that more than one personas can provide different level of information. Your answer should include 
all personas who can be the best persona to consult for the question. You will use the example provided 
in a form of Internal thought to find the answer for all questions.

Question: What are the most common failure modes for wind turbine gearboxes? This will help me identify the failure modes that the anomaly model should be able to detect. 
Please use (Internal thought).

(Internal thought): first we find out the list of candidate personas mentioned in System Prompt. 

We found two personas listed in system prompt: [Subject Matter Expert, Data Scientist]. 

First, let us evaluate first persona (Subject Matter Expert). The Subject Matter Expert has knowledge of the domain and can provide
 information about the common failure modes for wind turbine gearboxes. This question is best suited for a Subject Matter Expert.
 The sentiment for Subject Matter Expert is positive.

Next, we evaluate second persona (Data Scientist). This question is primarily related to the domain knowledge of
 wind turbine gearboxes and their failure modes. Therefore, Data Scientist is not the best persona to consult for this question. 
 The sentiment for Data Scientist is negative. 

Overall, Subject Matter Expert has positive sentiment.

Answer: The final answer is Subject Matter Expert. (TOKENSTOP)

"""

    # The asset description is "Valve - Hydraulic Operated - Isolation - Piston Type".
    def __init__(
        self,
        name: str,
        custom_config=None,
    ):
        self.name = name
        self.genai_config = dict(self.DEFAULT_CONFIG)
        if custom_config:
            self.genai_config.update(custom_config)

        self._genai_messages = defaultdict(list)
        stateful = False

        # agent 1
        self.DSAgent = GenAIChatClient(
            name="DS",
            description="Data Scientist",
            skill="building machine learning model, data analytics, python programming",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.DSSystemPrompt,
            stateful=stateful,
        )

        # agent 2
        self.SMEAgent = GenAIChatClient(
            name="SME",
            description="Subject Matter Expert",
            skill="Provide domain knowledge for a particular industrial assets and their working condition",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.SMESystemPrompt,
            stateful=stateful,
        )

        # agent 3
        self.SummarizeAgent = GenAIChatClient(
            name="Summarizer",
            description="Answer Summarizer",
            skill="generate summary of provided document, document summarization task",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.InfoSummaryPromt,
            stateful=stateful,
        )

        # agent 4
        self.QuestionGeneratorAgent = GenAIChatClient(
            name="QA",
            description="Question Answer Generator",
            skill="generate new set of questions from input documents",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.QuestionGenerator,
            stateful=stateful,
        )

        # agent 5
        self.QuestionClassifierAgent = GenAIInstructClient(
            name="QClassifier",
            description="Find right person for a given question",
            skill="use LLM to classifier a question into persona",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.QAClassifierSystemPrompt,
            question_message=self.QAClassifierPrompt,
        )

        # messages - storage
        self.genai_questions_for_sme = []
        self.genai_responses_from_sme = []
        self.genai_questions_for_ds = []
        self.genai_responses_from_ds = []
        self.total_processed_question_sme = 0
        self.total_processed_question_ds = 0

        self.testmode = 1

    def set_asset_class(self, asset_class):
        self.asset_class = asset_class

    def init_round(self, message, experiment_id):
        """Zero shot"""
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Round 1 ------------------------------.{Style.RESET_ALL}"
            )

        ds_response = self.DSAgent.create(
            messages=[{"content": message, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )

        sme_response = self.SMEAgent.create(
            messages=[{"content": message, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )

        # extract initial set of questions prepared by DS
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} ds_response >>> {Style.RESET_ALL}")
            print(ds_response)
            print(f"{Style.BRIGHT}{Fore.BLUE} sme_response >>> {Style.RESET_ALL}")
            print(sme_response)

        ds_questions = self.DSAgent.extract_questions(ds_response)
        f_ds_question = filter_and_sort_questions_using_reference(
            ds_questions,
            ds_questions,
            is_identical=True,
            filter_threshold=0.98,
        )
        self.genai_questions_for_sme.extend(f_ds_question)
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} ds_questions : {len(ds_questions)} >>> {Style.RESET_ALL}"
            )
            print(ds_questions)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} f_ds_question {len(f_ds_question)} >>> {Style.RESET_ALL}"
            )
            print(f_ds_question)

        """In Context Learning : Invoke Summarize Agent and then Get summary"""
        ds_summary = self.SummarizeAgent.create(
            messages=[{"content": sme_response + "\n Generate one paragraph summary.", "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        self.context_documents_ = ds_summary
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} ds_summary : {len(ds_summary)} >>> {Style.RESET_ALL}"
            )
            print(ds_summary)

        """ Now ask DS """
        addon = (
            ds_summary
            # + "\n\n Would you like to add additional set of questions based on provided information?"
            + "\n\n From the above summary, generate few more questions to be asked to subject matter expert."
        )
        ds_response_1 = self.DSAgent.create(
            messages=[{"content": addon, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )

        ds_questions_1 = self.DSAgent.extract_questions(ds_response_1)
        f_ds_question_1 = filter_and_sort_questions_using_reference(
            self.genai_questions_for_sme, ds_questions_1, filter_threshold=0.98
        )
        f_ds_question_2 = filter_and_sort_questions(
            f_ds_question_1, filter_threshold=0.98
        )
        self.genai_questions_for_sme.extend(f_ds_question_2)
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} ds_response_1 : {len(ds_response_1)} >>> {Style.RESET_ALL}"
            )
            print(ds_response_1)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} ds_questions_1 : {len(ds_questions_1)} >>> {Style.RESET_ALL}"
            )
            print(ds_questions_1)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} f_ds_question_2 : {len(f_ds_question_2)} >>> {Style.RESET_ALL}"
            )
            print(f_ds_question_2)
        self.context_questions_seeds_total_ = len(self.genai_questions_for_sme)


        # now we generate context document
        result = "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(self.genai_questions_for_sme)]
        )
        result += "\n Generate one paragraph summary."
        ds_context_summary = self.SummarizeAgent.create(
            messages=[{"content": result, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        self.question_context_documents_ = ds_context_summary


    def next_round(self, experiment_id):
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Round 2 ------------------------------.{Style.RESET_ALL}"
            )

        """This is a round 2 - Where DS and SME talk to each other"""
        for item in self.genai_questions_for_sme:
            sme_response = self.SMEAgent.create(
                messages=[{"content": item, "role": "user"}],
                context=None,
                experiment_id=experiment_id,
            )
            if self.testmode:
                print(f"{Style.BRIGHT}{Fore.BLUE} Question : >>> {Style.RESET_ALL}")
                print(item)
                print(f"{Style.BRIGHT}{Fore.BLUE} Answer : >>> {Style.RESET_ALL}")
                print(sme_response)

            self.genai_responses_from_sme.append(sme_response)
            # print("<<<<<<<<<<<<<<<<<<<-------End--------->>>>>>>>>")

    def question_generation(self, experiment_id):
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Round 3 ------------------------------.{Style.RESET_ALL}"
            )

        tmp_DSets = []
        """This is a question generation"""
        # purelly using questions

        # approch 1. Q1, Q2, Q3, ..... , Q20 ---> Q21, ...., Q30 (Question Prediction)
        # we use the questions that were designed to ask SME

        # approach 1
        # all questions and let context to cut it
        randomized_questions = random.sample(
            self.genai_questions_for_sme, len(self.genai_questions_for_sme)
        )
        result = "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(randomized_questions)]
        )
        question_response = self.QuestionGeneratorAgent.create(
            messages=[{"content": result, "role": "user", "type": "qq"}],
            context=None,
            experiment_id=experiment_id,
        )
        question_response_1 = self.QuestionGeneratorAgent.extract_questions(
            question_response
        )
        tmp_DSets.extend(question_response_1)
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}")
            print(result)
            print(f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}")
            print(question_response)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} question_response_1 : {len(question_response_1)} >>> {Style.RESET_ALL}"
            )
            print(question_response_1)

        # approach 2
        # most recent first
        result = "\n".join(["1. " + self.genai_questions_for_sme[-1]])
        question_response = self.QuestionGeneratorAgent.create(
            messages=[{"content": result, "role": "user", "type": "qq"}],
            context=None,
            experiment_id=experiment_id,
        )
        question_response_2 = self.QuestionGeneratorAgent.extract_questions(
            question_response
        )
        tmp_DSets.extend(question_response_2)
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}")
            print(result)
            print(f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}")
            print(question_response)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(question_response_1)} >>> {Style.RESET_ALL}"
            )
            print(question_response_2)

        # approach 3 - random sampling
        # most recent first
        selected_elements = random.sample(
            self.genai_questions_for_sme, min(len(self.genai_questions_for_sme), 10)
        )
        result = "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(selected_elements)]
        )
        question_response = self.QuestionGeneratorAgent.create(
            messages=[{"content": result, "role": "user", "type": "qq"}],
            context=None,
            experiment_id=experiment_id,
        )
        question_response_3 = self.QuestionGeneratorAgent.extract_questions(
            question_response
        )
        tmp_DSets.extend(question_response_3)
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}")
            print(result)
            print(f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}")
            print(question_response)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(question_response_1)} >>> {Style.RESET_ALL}"
            )
            print(question_response_3)

        # approach 4. Q1, A1 --> Q2
        # purely using question-answer pair
        for qid in range(len(self.genai_questions_for_sme)):
            intermediate_result = f"Question: {self.genai_questions_for_sme[qid]} \n Answer: {self.genai_responses_from_sme[qid]}"
            result = self.SummarizeAgent.create(
                messages=[{"content": intermediate_result, "role": "user"}],
                context=None,
                experiment_id=experiment_id,
            )
            result = (
                result
                + "\n\n From the above summary, generate few more questions to be asked to Subject matter expert."
            )
            question_response = self.QuestionGeneratorAgent.create(
                messages=[{"content": result, "role": "user", "type": "qa"}],
                context=None,
                experiment_id=experiment_id,
            )
            question_response_1 = self.QuestionGeneratorAgent.extract_questions(
                question_response
            )
            tmp_DSets.extend(question_response_1)
            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input_1 :  >>> {Style.RESET_ALL}"
                )
                print(intermediate_result)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_summary  :  >>> {Style.RESET_ALL}"
                )
                print(result)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input : {len(question_response_1)} >>> {Style.RESET_ALL}"
                )
                print(question_response)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(question_response_1)} >>> {Style.RESET_ALL}"
                )
                print(question_response_1)

        f_tmp_DSets_1 = filter_and_sort_questions_using_reference(
            self.genai_questions_for_sme, tmp_DSets, filter_threshold=0.98
        )
        f_tmp_DSets_2 = filter_and_sort_questions(f_tmp_DSets_1, filter_threshold=0.98)
        self.genai_questions_for_sme.extend(f_tmp_DSets_2)

        print(len(self.genai_questions_for_sme))
        df = pd.DataFrame({"questions": self.genai_questions_for_sme})
        df.to_csv("genai_questions.csv", index=False)

    def question_assignment(self, experiment_id):
        """_summary_

        :param experiment_id: _description_
        :type experiment_id: _type_
        """

        # for each question
        # find whi will asnwer the questions
        # assign them into their respective queue

        # sindex = answer.rfind('Answer:')
        # eindex = answer.rfind('(TOKENSTOP')
        # print (answer[sindex+7:eindex])   # 7 = len('Answer:')
        answer_text = ""

        if "Subject Matter Expert" in answer_text:
            print("Add to SME")

        if "Data Scientist" in answer_text:
            print("Add to DS")

        pass

    def print_token_usage(self):
        self.DSAgent.print_token_usage()
        self.SMEAgent.print_token_usage()

    def init_chat(self):
        """_summary_"""

        # this is a context prompt
        self.context_prompt_ = 'The industrial asset class is ' + self.asset_class

        # setting the MLFLow experiments
        experiment_name = "MyExperiment_" + str(uuid.uuid4())
        experiment_id = mlflow.create_experiment(experiment_name)

        print(f">>> Message: {self.context_prompt_}")
        print(f">>> Experiment id: {experiment_id}")
        print(f">>> Experiment name: {experiment_name}")

        # start recording
        with mlflow.start_run(experiment_id=experiment_id):
            # Initial round
            self.init_round(message=self.context_prompt_, experiment_id=experiment_id)
            # if number of questions are Zero, do some post analysis.... else move forward
            # Now use initial seed questions for second round

            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.MAGENTA} CP: <<< {self.context_prompt_} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} CPDoc: <<< {self.context_documents_} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.CYAN} SeedQuestions: <<< {self.context_questions_seeds_total_} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.GREEN} CPDSDOC: <<< {self.question_context_documents_} >>> {Style.RESET_ALL}"
                )

            exit(0)

            if len(self.genai_questions_for_sme) > 0:
                # interact with SME and then talk to DS
                self.next_round(experiment_id=experiment_id)

                # now we have response from sme, so we can create few more questions
                self.question_generation(experiment_id=experiment_id)

                # now we can do question assignment
                self.question_assignment(experiment_id=experiment_id)

                # ideally we should have answer generation
                # self.answer_generation()

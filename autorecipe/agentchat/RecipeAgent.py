from autorecipe.genai.GenAIChat import GenAIChatClient
from collections import defaultdict
import mlflow


class RecipeAgent:
    # configuration
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
about the asset class such as wind turbine, pump, oil well, etc.
"""

    InfoSummaryPromt = """
Prepare a human-readable summary in a well-structured paragraph, eliminating any 
 special characters such as new lines and tabs. Focus on capturing the main components, failure modes and 
 reasons, degradation mechanisms, severity and detectability ratings, emphasizing the crucial insights for 
 predicting and preventing failures. Ensure the summary provides a coherent narrative. 
"""

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
question and you will select a persona who can answer the given question.  Your selection is based 
on the persona's field experience and scientific knowledge. 

Persona: Data Scientist
Skill: building machine learning model, data analytics, python programming

Persona: Subject Matter Expert
Skill: Provide domain knowledge for a particular industrial assets and their working condition
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
#The asset description is "Valve - Hydraulic Operated - Isolation - Piston Type".
    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.genai_config = self.DEFAULT_CONFIG.copy()
        self._genai_messages = defaultdict(list)

        # this is a data scien
        self.DSAgent = GenAIChatClient(
            name="DS",
            description='Data Scientist',
            skill='building machine learning model, data analytics, python programming',
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.DSSystemPrompt,
        )

        self.SMEAgent = GenAIChatClient(
            name="SME",
            description='Subject Matter Expert',
            skill='Provide domain knowledge for a particular industrial assets and their working condition',
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.SMESystemPrompt,
        )

        self.SummarizeAgent = GenAIChatClient(
            name="Summarizer",
            description='Answer Summarizer',
            skill='generate summary of provided document, document summarization task',
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.InfoSummaryPromt,
        )

        self.QuestionGeneratorAgent = GenAIChatClient(
            name="QA",
            description='Question Answer Generator',
            skill='generate new set of questions from input documents',
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            credentials=self.genai_config["creds"],
            system_message=self.QuestionGenerator,
        )

        self.genai_questions_for_sme = []
        self.genai_responses_from_sme = []
        self.genai_questions_for_ds = []

    def init_round(self, message, experiment_id):
        """This is a round 1"""
        ds_response = self.DSAgent.create(
            messages=[{"content": message, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        print(ds_response)

        sme_response = self.SMEAgent.create(
            messages=[{"content": message, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        ds_questions = self.DSAgent.extract_questions(ds_response)
        self.genai_questions_for_sme.extend(ds_questions)
        print(len(self.genai_questions_for_sme))

        # round 1 revision
        ds_summary = self.SummarizeAgent.create(
            messages=[{"content": sme_response, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        addon = (
            ds_summary
            + " Would you like to add additional set of questions based on provided information?"
        )
        ds_response_1 = self.DSAgent.create(
            messages=[{"content": addon, "role": "user"}],
            context=None,
            experiment_id=experiment_id,
        )
        ds_questions_1 = self.DSAgent.extract_questions(ds_response_1)
        self.genai_questions_for_sme.extend(ds_questions_1)
        print(len(self.genai_questions_for_sme))

    def next_round(self, experiment_id):
        """This is a round 2"""
        for item in self.genai_questions_for_sme:
            print("--------------------------Question--------->>>>>>>>>")
            print(item)
            sme_response = self.SMEAgent.create(
                messages=[{"content": item, "role": "user"}],
                context=None,
                experiment_id=experiment_id,
            )
            print("--------------------------Answer--------->>>>>>>>>")
            print(sme_response)
            self.genai_responses_from_sme.append(sme_response)
            print("<<<<<<<<<<<<<<<<<<<-------End--------->>>>>>>>>")

    def question_generation(self, experiment_id):
        """This is a question generation"""
        # purelly using questions

        # approch 1. Q1, Q2, Q3, ..... , Q20 ---> Q21, ...., Q30 (Question Prediction)
        result = "\n".join(
            ["question: " + item for item in self.genai_questions_for_sme]
        )
        question_response = self.QuestionGeneratorAgent.create(
            messages=[{"content": result, "role": "user", "type": "qq"}],
            context=None,
            experiment_id=experiment_id,
        )
        print(question_response)

        # approach 2. Q1, A1 --> Q2
        # purely using question-answer pair
        for qid in range(len(self.genai_questions_for_sme)):
            result = f"Question: {self.genai_questions_for_sme[qid]} \n Answer: {self.genai_responses_from_sme[qid]}"
            question_response = self.QuestionGeneratorAgent.create(
                messages=[{"content": result, "role": "user", "type": "qa"}],
                context=None,
                experiment_id=experiment_id,
            )
            print(question_response)

    def print_token_usage(self):
        self.DSAgent.print_token_usage()
        self.SMEAgent.print_token_usage()

    def init_chat(self, message):
        """_summary_"""
        import uuid
        experiment_name = "MyExperiment_" + str(uuid.uuid4())
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f">>> Message: {message}")
        print(f">>> Experiment id: {experiment_id}")
        print(f">>> Experiment name: {experiment_name}")
        with mlflow.start_run(experiment_id=experiment_id):
            self.init_round(message=message, experiment_id=experiment_id)
            self.print_token_usage()
            self.next_round(experiment_id=experiment_id)
            self.print_token_usage()
            self.question_generation(experiment_id=experiment_id)
            self.print_token_usage()

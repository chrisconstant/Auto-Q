from autorecipe.genai.GenAIChat import GenAIChatClient
from collections import defaultdict

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
"""

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

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.genai_config = self.DEFAULT_CONFIG.copy()
        self._genai_messages = defaultdict(list)

        self.DSAgent = GenAIChatClient(
            model = self.genai_config["model"],
            params = self.genai_config["params"],
            credentials = self.genai_config["creds"],
            system_message = self.DSSystemPrompt,
        )

        self.SMEAgent = GenAIChatClient(
            model = self.genai_config["model"],
            params = self.genai_config["params"],
            credentials = self.genai_config["creds"],
            system_message = self.SMESystemPrompt,
        )


        self.SummarizeAgent = GenAIChatClient(
            model = self.genai_config["model"],
            params = self.genai_config["params"],
            credentials = self.genai_config["creds"],
            system_message = self.InfoSummaryPromt,
        )

        self.genai_questions_for_sme = []
        self.genai_responses_from_sme = []


    def init_round(self, message):
        """This is a round 1
        """   
        ds_response = self.DSAgent.create(messages=[{'content': message, 'role': 'user'}], context=None)
        sme_response = self.SMEAgent.create(messages=[{'content': message, 'role': 'user'}], context=None)
        ds_questions = self.DSAgent.extract_questions(ds_response)
        self.genai_questions_for_sme.extend(ds_questions)

        # round 1 revision
        ds_summary = self.SummarizeAgent.create(messages=[{'content': sme_response, 'role': 'user'}], context=None)
        addon = ds_summary + ' Would you like to add additional set of questions based on provided information?'
        ds_response_1 = self.DSAgent.create(messages=[{'content': addon, 'role': 'user'}], context=None)
        ds_questions_1 = self.DSAgent.extract_questions(ds_response_1)
        self.genai_questions_for_sme.extend(ds_questions_1)

    def next_round(self):
        """This is a round 2
        """   
        for item in self.genai_questions_for_sme:
            sme_response = self.SMEAgent.create(messages=[{'content': item, 'role': 'user'}], context=None)
            print (sme_response)
            self.genai_responses_from_sme.append(sme_response)

    def init_chat(self, message):
        """_summary_
        """
        self.init_round(message=message)
        self.next_round()
        


from autorecipe.genai.GenAIChat import GenAIChatClient
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationReturnOptions,
)
import uuid
import mlflow
import ray

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
creds = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

SystemPrompt = """User is working on knowledge extraction task using question-answering
approach. User has provided an industrial asset name and a asset description. User
has generated several questions and answer of these questions can have information 
about several aspects of asset. User is focused on a particular aspect or topic for
a knowledge extraction task.

Asset Name: {assetname}

Asset Description: {assetdescription}

knowledge extraction focus: {knowledgeextraction}

User will provide a question and your task is to check wether answer of the question contain 
the information user is looking for as specified in knowledge extraction focus. Note that, question 
may not directly ask information about knowledge extraction focus or focused on that aspect but 
answer of the question may provide information. In such situation your answer is Yes. It is possible 
that question is not related to the information user is looking for. In such case, return No Answer.  

"""

assetname = """Pulverizer - Coal - Roll Wheel Type"""
assetdescription = """A pulverizer, specifically a coal pulverizer, is a mechanical device that grinds 
coal into a fine powder. The coal is fed into the mill through a central inlet pipe, where it is surrounded 
by grinding elements in the form of wheels or rollers. As the mill rotates, the grinding elements crush the 
coal, turning it into a powder. The pulverized coal is then transported to the boiler where it is burned 
to generate steam. The main components of a coal pulverizer include the grinding table, grinding rollers 
(also known as roll wheels), hydraulic system, classifier, and exhaust system. The grinding table is the 
top part of the mill where the coal is fed into. The grinding rollers are the rolling elements that crush 
the coal. The hydraulic system provides the force needed to rotate the grinding rollers. The classifier is 
used to separate the fine powder from the coarse powder, and the exhaust system removes the dust generated 
during the grinding process."""

knowledgeextraction = """User is interested in extracting failure locations while conducting
 failure mode and effect analysis (FMEA)."""

LLMsets = ['ibm/granite-13b-chat-v2',
        'meta-llama/llama-2-70b-chat',
        'ibm/granite-13b-labrador-rc',
        'ibm-mistralai/mixtral-8x7b-instruct-v01-q'
        ]

params = {
    "decoding_method": DecodingMethod.GREEDY,
    "min_new_tokens": 100,
    "max_new_tokens": 500,  # 1500,
    "stop_sequences": ["(TOKENSTOP)"],
    "return_options": TextGenerationReturnOptions(input_text=False, input_tokens=True),
    "moderations": ModerationParameters(
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
}

experiment_name = "MyExperiment_" + str(uuid.uuid4())
experiment_id = mlflow.create_experiment(experiment_name)

ray.init()

#@ray.remote
def run_one(assetname, 
            assetdescription, 
            knowledgeextraction, 
            fc = 'What are the failure mode of Pulverizer - Coal?',
            model_id=0):
    """
    pass the topic
    """
    ansSet = []
    Sprompt = SystemPrompt.format(assetname=assetname,
                                  assetdescription=assetdescription,
                                  knowledgeextraction=knowledgeextraction)

    tmpClient = GenAIChatClient(name='Testing',
                                    description='I am testing classifier',
                                    skill='Generic',
                                    model=LLMsets[model_id],
                                    params=params,
                                    credentials=creds,
                                    system_message=Sprompt,
                                    stateful=True)

    fcode_template = """
    User has following question: 

    Question: {fc}

    Does this Question is relevant to the knowledge extraction focus area as specified in system prompt?

    """

    fccode = fcode_template.format(fc=fc)
    answer = tmpClient.create(
        context=None,
        messages=[{"content": fccode, "role": "user"}],
        experiment_id=experiment_id,
    )
    answer = answer.strip()
    answer = answer.split('\n\n')[0]
    print (answer)

fcSet = [
    'Can you describe the failure modes associated with the hydraulic system in a coal pulverizer?',
    'Can you describe the failure modes associated with the grinding table in a coal pulverizer?',
    'Can you describe the failure modes associated with the exhaust system in a coal pulverizer?',
    'Can you describe the failure modes associated with the classifier in a coal pulverizer?',
    'What are the potential causes of failure for the grinding rollers in a coal pulverizer?',
    'What are the potential causes of failure for the grinding table in a coal pulverizer?',
    'What are the potential causes of failure for the exhaust system in a coal pulverizer?',
]

for fc in fcSet:
    run_one(assetname, assetdescription, knowledgeextraction, fc=fc, model_id=2)

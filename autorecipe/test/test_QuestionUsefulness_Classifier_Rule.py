import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schemas import ChatOptions, GenerateParams, ReturnOptions
from genai.schemas.generate_params import HAPOptions, ModerationsOptions

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
api_key = "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg"
api_endpoint = "https://bam-api.res.ibm.com"

SystemPrompt = """
System Prompt: You are a helpful, respectful, and honest assistant. User will provide list of questions written by 
various different persona. These questions are labelled as useful question. Your role is to provide rules that justify 
why these questions are labelled as useful.

"""

ContextPrompt = """Question: Here is a list of questions labelled as "useful":
1. where would one find out the tax bracket rates for their country ?
2. hey , is n't the price and hence demand predetermined by the industry ?
3. what about 'investing ' in a ponzi scheme and withdrawing in the early days before it busts ?
4. when would the value of a product increase ?
5. at 8:50 when grandmother deposited the cash in the banks a/c, why did only the reserve rise , why not the liabilities and hence the total balance sheets of the bank ?
6. a person in the market who expect that the prices of products will increase in future is known as ?
7. why do people from forex always advertise and want people to exchange what benefits do they gain ?
8. can a bank be a member of more than one network ?
9. if i have a 15 year fixed mortgage , it means i have a 15 year amortization ?
10. do the fixed rates adjust to inflation over their periods ?
Please use (Internal thought).

(Internal thought).

Let me think step by step.

(justification). Here is a list of rules that justify why above questions are labelled as useful. 
1. This question is useful because it helps people understand how to find tax bracket rates, which is important for tax planning and financial decision-making.
2. This question is useful because it helps people understand the relationship between price, demand, and the industry, which is important for businesses to make informed decisions.
3. This question is useful because it helps people understand the risks and potential benefits of investing in a ponzi scheme, which is important for making informed investment decisions.
4. This question is useful because it helps people understand when the value of a product might increase, which is important for businesses to make informed pricing decisions.
5. This question is useful because it helps people understand how bank reserves and balance sheets work, which is important for understanding the banking system and making informed financial decisions.
6. This question is useful because it helps people understand the definition of a person who expects prices to increase in the future, which is important for understanding market trends and making informed investment decisions.
7. This question is useful because it helps people understand why forex advertisements are common and what benefits they offer, which is important for making informed investment decisions.
8. This question is useful because it helps people understand whether a bank can be a member of multiple networks, which is important for understanding banking operations and making informed financial decisions.
9. This question is useful because it helps people understand the relationship between a 15-year fixed mortgage and a 15-year amortization, which is important for understanding mortgage options and making informed financial decisions.
10. This question is useful because it helps people understand how fixed rates adjust to inflation over time, which is important for understanding the impact of inflation on financial decisions.

Next, the above justification is question specific. I will generalize these rules across questions and avoid any domain specific 
information. The rules should be domain agnostic, i.e., can be applicable to label question from any other domain. 

Let me think step by step.

(domain-agnostic rules). Here are the domain-agnostic rules that justify why the above questions are labelled as useful:
1. Questions that provide information or resources that can be used to make informed decisions are useful.
2. Questions that help people understand relationships between different elements in a system are useful.
3. Questions that help people understand risks and potential benefits of a particular action or investment are useful.
4. Questions that help people understand when a product's value might increase are useful.
5. Questions that help people understand how a system or process works are useful.
6. Questions that provide definitions or explanations of terms are useful.
7. Questions that help people understand the benefits of a particular service or product are useful.
8. Questions that help people understand whether a particular entity can be part of multiple groups or networks are useful.
9. Questions that help people understand the relationship between different financial products or options are useful.
10. Questions that help people understand how different factors, such as inflation, affect financial decisions are useful.

Finally, I will create a concise and human-readable version of the above domain-agnostic rules.

Let me think step by step.

(concise rules). Here are the concise and human-readable version of the above domain-agnostic rules:
1. Questions that seek information or resources to make informed decisions are useful.
2. Questions that seek explanations of relationships between different elements are useful.
3. Questions that seek explanations of risks and potential benefits are useful.
4. Questions that seek explanations of when a product's value might increase are useful.
5. Questions that seek explanations of how a system or process works are useful.
6. Questions that seek definitions or explanations of terms are useful.
7. Questions that seek explanations of benefits of a service or product are useful.
8. Questions that seek explanations of whether a particular entity can be part of multiple groups or networks are useful.
9. Questions that seek explanations of the relationship between different financial products or options are useful.
10. Questions that seek explanations of how different factors affect financial decisions are useful.

Answer: Here are final rules:
1. Information-seeking questions: Questions that request information or resources to make informed decisions are useful.
2. Relationship-seeking questions: Questions that seek explanations of relationships between different elements are useful.
3. Risk-benefit questions: Questions that seek explanations of risks and potential benefits are useful.
4. Value-increase questions: Questions that seek explanations of when a product's value might increase are useful.
5. Process-seeking questions: Questions that seek explanations of how a system or process works are useful.
6. Definition-seeking questions: Questions that seek definitions or explanations of terms are useful.
7. Benefit-seeking questions: Questions that seek explanations of benefits of a service or product are 
(TOKENSTOP)
"""


Input1 = """Question:Here is a list of questions labelled as "useful":
1. if christ was circumcised how come christian men ( unlike jews and muslims ) are not required to be circumcised ?
2. is world war 2 still going on because when the soviet union declared war on japan they did not sign a treaty ?
3. who were the leaders of every country ?
4. how much did the viking mission cost ?
5. was the picture of mussolini of when he gathered followers for his fascist party ?
6. and what is the rule by which i can replace amolecule by another in its position ?
7. why urbanization in indus valley civilization happened faster than egyptian and mesopotamian civilizations ?
8. was the german blitzkrieg strategy in world war ii based upon the schlieffen plan at all ?
9. who is `` hans hillewaert '' ?
10. how much too late were the germans to realize that they could outflank the french ?
Please use (Internal thought).

"""

Input2 = """Question:Here is a list of questions labelled as "useful":
1. and with a double flute or a triple flute ( both of which are still existent ) would n't you get an interval or a triad ( 3 note chord ) on 1 flute ?
2. why does salt lower the freezing point ?
3. what does the value of spring constant `` k '' depend upon ?
4. the letter n represents a number , between 50 and 60. the gcf of n and 16 is 8 , find the n ?
5. is that correct that the cross product of the span of n ( a ) is always part of c ( at ) ?
6. if hundred is divided by 3 what will be the exact answer without remainder ?
7. if you picked a random number 0 through infinity would it be just infinity or some big number ?
8. so , if i did n't do very well in my 9th and 10th grade classes ( typically c+ to b+ except in language where i got a 99 average ) will i still be okay if i do well in my junior and senior years ?
9. @ 6:10 so even if the carbon next to the chiral center has a f plus 2 ch3 groups attached to it , we still consider the c attached to the br as the higher one ?
10. does the hybridization have a 67 % mix of s characteristic and 33 % p characteristic ?
Please use (Internal thought).

"""

Input3 = """Question:Here is a list of questions labelled as "useful":
1. can someone give me more information on cuneiform ?
2. how do you create a server when creating a webpage not on khan ?
3. precisely speaking , how is computer programming linked to binary numbers ( the language of a computer ) to produce software for computers and other electronic devices ?
4. how is the internet linked to you broadband , are they moderators ?
5. does n't streaming typically utilize udp , rather than tcp , since the connection does n't necessarily have to be constantly acknowledged ?
6. how can we learn to be an ethical hacker ?
7. do we always get assigned to the same cookie by a specific website like tumblr ?
8. who is the creator of web ?
9. which programming language does khanacademy use ?
10. why is there a fingerprint on the photo ?
Please use (Internal thought).

"""

ExplanationPrompt = """Here is a list of rules that justify why above questions are labelled as useful. The justification is question specific. 
Can you generalize these rules across questions. The rules should be domain agnostic, i.e., can be applicable to label
question from any other domain.

"""

ExtendedExplanationPrompt = """

Let me think step by step.
"""

test = """Can you generate some rule on why these questions are labelled as useful?

Answer: Here are some examples of guideline that define what is usefulness of question means. 
Note that these guidelines are not complete and you should suggest any other guidelines. 
1. Questions that are relevant, informative, and thought-provoking in the context of particular domain 
are candidates for "usefulness". 
2. Question is useful if it has a level of depth and complexity that makes them useful for learning and discussion. 
3. The questions are clear and concise, making it easy for others to understand and respond to them.
4. The questions generate curiosity and desire to understand specific concepts and principles.
5. The questions that are open-ended and encourage exploration and discovery are useful.
6. The questions encourage investigation and research, prompting individuals to explore new ideas and concepts.
7. 


"""

# They are relevant, clear, concise, curious, open-ended, and encouraging exploration and discovery.
# They prompt individuals to investigate and research new ideas and concepts related to various domains


LLMsets = [
    # "ibm/granite-13b-instruct-v2",
    "meta-llama/llama-2-70b-chat",
    # "google/flan-ul2",
    "thebloke/mixtral-8x7b-instruct-v0-1-gptq",
]

# Is this question for Subject Matter Expert?
# Is this question for Data Scientist?

for i in range(len(LLMsets)):
    llm = LangChainInterface(
        model=LLMsets[i],
        credentials=Credentials(api_key, api_endpoint),
        params=GenerateParams(
            decoding_method="greedy",
            max_new_tokens=2000,
            min_new_tokens=200,
            temperature=0.5,
            top_k=50,
            top_p=1,
            stream=True,
            stop_sequences=["(TOKENSTOP)"],
            return_options=ReturnOptions(input_text=False, input_tokens=True),
            moderations=ModerationsOptions(
                # Threshold is set to very low level to flag everything (testing purposes)
                # or set to True to enable HAP with default settings
                hap=HAPOptions(input=True, output=False, threshold=0.01)
            ),
        ),
    )

    # step 1
    print(
        "Start...-----------------------------------------------------------------------------"
    )
    answer = llm.generate(
        prompts=[f"System Prompt: {SystemPrompt} \n\n {ContextPrompt} \n\n {Input3}"]
    )
    print (answer.generations[0][0].text)
    print(
        "-----------------------------------------------------------------------------...End"
    )

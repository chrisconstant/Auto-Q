from autorecipe import AssistantAgent
from autorecipe import UserProxyAgent

Message = """
You are a helpful AI assistant. You act as a data scientist. Solve tasks using your coding and language skills. 
Your job is to build an anomaly model using real time time series sensor data
 obtained from IoT/OT system. In order to get domain understanding of the problem you will prepare a series of 
 questions to be asked in sequential orders to subject matter experts. Typical questions should focus on the
important components for which anomaly model should be build, the important failure modes and the ability of
 sensor data to detect these failure. 
"""

Message1 = """
You act as a data scientist. Your job is to build an anomaly model using real time time series sensor data
 obtained from IoT/OT system. In order to get domain understanding of the problem you will prepare a series of 
 questions to be asked in sequential orders to subject matter experts. Typical questions should focus on the
important components for which anomaly model should be build, the important failure modes and the ability of
 sensor data to detect these failure. Also include questions that focus on preparing meta data information for sensor 
 variable along with their units and operating range. Be specific about the information you want: Instead of 
 asking a general question like "What are the most common failure modes in wind turbines?", try to ask a more 
 specific question like "What are the three most common failure modes in wind turbine bearings, and what are 
 the typical symptoms of each failure mode?
"""


# What are the three most common failure modes in wind turbine bearings, and what are the typical symptoms of each failure mode?
# What are some common environmental factors that can impact wind turbine performance, such as temperature, humidity, or air quality
# What are the three most common failure modes in wind turbine bearings, and what are the typical symptoms of each failure mode? Please provide specific examples of each failure mode and its symptoms, and explain how these failure modes can impact the system\'s performance

proxAgt = UserProxyAgent(name="Agent")
LLMAgt = AssistantAgent(name="Chatbot", system_message=Message)

proxAgt.initiate_chat(
    LLMAgt,
    message="""The industrial asset class is wind turbine""",
)


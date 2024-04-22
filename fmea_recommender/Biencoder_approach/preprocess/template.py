# this is a template to create a query

user2item_template = [
    "I have the following histories: {}.",
]

description2label_template = [
    "I have the following label description: {}",
    "Recommend a class label for me based on label description: {}.",
    "Given the following label description {}, predict the most possible associated class label",
    "Find a class label for me based on the label description: {}.",
    "Class description include: {}.",
    "Can you classify the following label description: {}?",
    "What class label would you assign to the given label description: {}?",
    "Based on the label description: {}, suggest a class label.",
    "Predict the most likely class label for the given label description: {}.",
    "Can you determine the appropriate class label for the following description: {}?",
    "Given the label description: {}, recommend a suitable class label.",
    "What class would you assign to the following label description: {}?",
    "Can you suggest a class label based on the given label description: {}?",
    "What is the most probable class label for the label description: {}?",
    "Can you identify the class label for the given label description: {}?",
]

example2label_template = [
    "I have the following example to classify: {}",
    "Recommend a class label for me based on example description: {}.",
    "Given the following description {}, predict the most possible associated class label",
    "Find a class label for me based on the description: {}.",
    "example description include: {}.",
    "Can you classify the following example: {}?",
    "What class label would you assign to the given example description: {}?",
    "Based on the description: {}, suggest a class label.",
    "Predict the most likely class label for the given example description: {}",
    "Can you determine the appropriate class label for the following description: {}?",
    "Given the example description: {}, recommend a suitable class label.",
    "What class would you assign to the following example: {}?",
    "Can you suggest a class label based on the given example description: {}?",
    "What is the most probable class label for the example description: {}?",
    "Can you identify the class label for the given example: {}?",
]

example2components_template = [
    "Which components can experience the following failure {}?",
    "Find the components for me that are associated with the given description : {}?",
    "Which parts of a system can be affected by {}?",
    "Can you specify the components associated with the issue of {}?",
    "Which system elements can be subject to the problem of {}?",
    "What are the components that can be influenced by the failure of {}?",
    "Can you name the parts that can encounter the problem of {}?",
    "Which system components can be affected by the failure of {}?",
    "Which parts of a system are susceptible to the failure of {}?",
    "Can you identify the components associated with the issue of {}?",
    "What components are prone to experiencing the problem of {}?",
]

example_equipment_component2label_template = [
    "I want to find a class label with the following description: {0} for an equipment category {1}. These are the components of equipment that experience the given failure : {2}.",
    "Find a class label that describes a type of equipment failure that affects the {1} category of equipment. The failure is characterized by {0}, and the components that are typically affected are {2}",
    "I'm looking for a class label that corresponds to a specific equipment failure mode. The description is {0}, and it impacts the {1} category of equipment. The components of the equipment that are affected by this failure are {2}",
    "find a class label that describes a failure mode that occurs in the {1} category of equipment. The failure is described as {0}, and the affected components are {2}",
    "Could you help me find a class label for an equipment category failure? The description is: {0} for the equipment category {1}. The components that usually experience this failure are: {2}.",
    "I'm looking for a class label that represents a type of equipment failure in the {1} category. The failure is defined by: {0}, and the components that are typically affected are: {2}",
    "I need assistance in finding a class label for a specific equipment failure mode. The failure mode is: {0}, and it affects the {1} category of equipment. The components that are impacted by this failure are: {2}",
    "Can you help me find a class label for a failure mode in the {1} category of equipment? The failure is described as: {0}, and the affected components are: {2}.",
    "Find class label for {0} in equipment category {1}; affected components: {2}",
    "Class label for {1} equipment category failure: {0}, affected components: {2}."
]

example2description = [
    
]

# You have given a failure mode description: Leakage in closed position. You need to generate 10 reformulation of the given description.
# can you use this for additional reformulation: "sample": "Both pods are not functioning as desired", "asset_name": "drilling_equipment", "components": ["Subsea blowout preventers (BOP)"]
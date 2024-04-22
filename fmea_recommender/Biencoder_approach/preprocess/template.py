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

# You have given a failure mode description: Leakage in closed position. You need to generate 10 reformulation of the given description.
# can you use this for additional reformulation: "sample": "Both pods are not functioning as desired", "asset_name": "drilling_equipment", "components": ["Subsea blowout preventers (BOP)"]
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
]

example2label_template = [
    "I have the following example to classify: {}",
    "Recommend a class label for me based on example description: {}.",
    "Given the following description {}, predict the most possible associated class label",
    "Find a class label for me based on the description: {}.",
    "example description include: {}.",
]

example3components_template = [
    "Which components can experience the following given failure {}?",
    "Find the components for me that are associated with the given description: {}?",
]

# You have given a failure mode description: Leakage in closed position. You need to generate 10 reformulation of the given description.
# can you use this for additional reformulation: "sample": "Both pods are not functioning as desired", "asset_name": "drilling_equipment", "components": ["Subsea blowout preventers (BOP)"]
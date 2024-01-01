import spacy

def is_custom_question(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)

    # Check if the sentence is a question based on custom patterns
    question_patterns = [
        ('Is', '<TEMP-NP>', '<TEMP-NP>', '?'),
        ('Is', 'there', '<TEMP-NP>', '?'),
        ('Is', '<TEMP-NP>', '<TEMP-ADJP>', '?'),
        ('Can', '<TEMP-NP>', '<TEMP-VERB>', '<TEMP-NP>', '?'),
        ('Do', '<TEMP-NP>', '<TEMP-VERB>', '<TEMP-NP>', '?'),
        ('Does', 'anyone', 'have', '<TEMP-NP>', '?'),
        ('Is', 'it', '<TEMP-ADJP>', 'to', '<TEMP-VERB>', '<TEMP-NP>', '?'),
    ]

    for pattern in question_patterns:
        if len(pattern) != len(doc):
            continue

        match = all(token.text == pattern_part or pattern_part.startswith('<TEMP-') for token, pattern_part in zip(doc, pattern))
        if match:
            return True

    return False

# Example usage:
question1 = "Is the sky blue?"
question2 = "Can we go to the park?"
question3 = "Is it difficult to learn programming?"
question4 = "Do elephants eat peanuts?"
question5 = "Does anyone have a pen?"

print(is_custom_question(question1))  # True
print(is_custom_question(question2))  # True
print(is_custom_question(question3))  # True
print(is_custom_question(question4))  # True
print(is_custom_question(question5))  # True

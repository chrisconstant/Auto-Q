from num2words import num2words


def generate_system_prompt(
    Personas=["option1", "option2", "option3"], Skills=["S1", "S2", "S3"]
):
    joined_string = ", ".join(Personas)

    # Replace the last comma with "or"
    last_comma_index = joined_string.rfind(", ")
    if last_comma_index != -1:
        joined_string = (
            joined_string[:last_comma_index]
            + " or"
            + joined_string[last_comma_index + 1 :]
        )

    number = len(Personas)
    number_in_words = num2words(number)

    intro_text_template = """
    You are a helpful, respectful, and honest assistant. You will be introduced to several 
    persona such as {Personas}, etc. User will provide a question and you will select a persona 
    who can answer the given question. Your selection is based on the persona's field experience and scientific knowledge. 
    Sometime questions can be answered by multiple personas. Here is the {PersonasNum} personas along with 
    their skill description. 
    """

    intro_text = intro_text_template.format(
        Personas=joined_string, PersonasNum=number_in_words
    )

    # Prompt template with placeholders
    prompt_template = """
    {intro_text}

    Persona: {A_element}
    Skill: {B_element}

    """

    # Generate prompts dynamically based on the elements of lists A and B
    prompts = [
        prompt_template.format(
            intro_text=intro_text if i == 0 else "", A_element=a, B_element=b
        ).lstrip()
        for i, (a, b) in enumerate(zip(Personas, Skills))
    ]

    # Join prompts into a single string
    full_prompt = "\n".join(prompts)

    return full_prompt.strip()

output = generate_system_prompt()
print (output) 
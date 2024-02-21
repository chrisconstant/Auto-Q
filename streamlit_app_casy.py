# app.py
import streamlit as st
import matplotlib.pyplot as plt
import time

system_prompt = ''
ds_promt = ''
sme_prompt = ''

question = []
ds_answer = []
sme_answer = []
all_question = []

assets = ["Standby Generator", "Wind Turbine Gearbox", "Air Compressor", "Hydroelectric Power Turbine", "Electrical Transformer", "Induced Draft Fan", "Blast Furnace", "Electric Battery", "Substation Electrical Transformer","Water-Cooled Condenser","Industrial Robot", "Turbine Generator", "Industrial boiler", "Industrial Oven", "Industrial Furnace", "Centrifugal Compressor", "Hydraulic Press", "Steam Turbine", "Chiller", "Air Handling Unit"]

def display_message(display_message, tsleep=0.1):
    message_placeholder = st.empty()
    full_response = ""

    lines = display_message.splitlines()
    for line in lines:
        words = line.split()
        for chunk in words:
            full_response += chunk + " "
            time.sleep(tsleep)
            message_placeholder.markdown(full_response + "▌")
        full_response += "\n"

    message_placeholder.markdown(full_response)

def collect_system_messages(assetname='windturbinegearbox',model='mixtral'):
    """_summary_
    """
    global system_prompt, ds_promt, sme_prompt

    import pandas as pd 
    df = pd.read_csv(f'./results/{assetname}/genai_context_docs_{assetname}_{model}.csv')
    system_prompt = list(df['context_prompt'])[0]
    ds_promt = list(df['ds_context'])[0]
    sme_prompt = list(df['sme_context'])[0]
    
def sensor_failure_tags():
    """_summary_
    """
    global smePrompt, smeAnswer

    import pandas as pd
    df = pd.read_csv(f'./ijcairesults/all_prompts_answers_modified.csv')
    smePrompt = list(df['Prompt'])
    smeAnswer = list(df['Answer'])

def pattern_template_library():
    """_summary_
    """
    global tem_behaviour, pcode_ans, scode_ans, dmessage

    import pandas as pd
    df = pd.read_csv(f'./ijcairesults/final_answer.csv')
    tem_behaviour = list(df['tem_behaviour'])
    pcode_ans = list(df['pcode_ans'])
    scode_ans = list(df['scode_ans'])
    dmessage = list(df['dmessage'])

def trim_repeated_right_side(sentence, max_repetitions):
    import re
    # Define the regex pattern to match repeated occurrences of the last letter
    pattern = re.compile(rf'({re.escape(sentence[-1])})\1{{{max_repetitions - 1},}}$')

    # Trim the right side by replacing repeated occurrences with a single occurrence
    result_word = pattern.sub(sentence[-1], sentence)

    return result_word

def reduction_factor(sentence):
    sent1 = trim_repeated_right_side(sentence, max_repetitions=50)
    return len(sent1)*100.0/len(sentence)

def plot_results(total_questions = 6617, answered_questions = 488):
    unanswered_questions = total_questions - answered_questions

    unanswered_questions = total_questions - answered_questions

    # Calculate the percentage of answered and unanswered questions
    answered_percentage = (answered_questions / total_questions) * 100
    unanswered_percentage = (unanswered_questions / total_questions) * 100

    # Define the colors for the pie chart slices
    colors = ['lightblue', 'lightcoral']

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(2, 2))  # Adjust the figsize to make the pie chart smaller
    ax.pie([answered_percentage, unanswered_percentage], 
           labels=['Answered Questions', 'Unanswered Questions'], 
           autopct='%1.1f%%', startangle=90, colors=colors)

    ax.axis('equal')
    st.pyplot(fig)

def display_chat_message_warning(score):
    # Define emoji based on the score
    if score < 10.0:
        score_emoji = "👎 Ignore Answer Please"
        score_color = "red"
        # Display chat message with score
        st.markdown(f'<p style="color: {score_color};">{score_emoji}</p>', unsafe_allow_html=True)
    else:
        score_emoji = "👍 Initial Quality Check Passed"
        score_color = "green"
        # Display chat message with score
        st.markdown(f'<p style="color: {score_color};">{score_emoji}</p>', unsafe_allow_html=True)

def get_markdown(json_string, i=1):
    import json
    json_string = json_string.replace("'", '"')
    data = json.loads(json_string)
    if i == 1:
        markdown_string = "#### Equipment Failure Modes\n\n"
        markdown_string += "| Equipment | Failure Mode |\n| --- | --- |\n"

        # Iterate over dictionary items and format as Markdown table rows
        for key, value in data.items():
            for mode in value:
                markdown_string += f"| {key} | {mode} |\n"
    elif i == 3:
        markdown_string = "#### Sensor-Failure Association\n\n"
        markdown_string = '##### Component : ' + data['component'] + '\n'
        markdown_string = '##### Failure mode : ' + data['failure_mode'] + '\n'
        del data['component']
        del data['failure_mode']
        markdown_string += "| Sensor | Temporal Behavior | Description |\n| --- | --- | --- |\n"
        for key, value in data.items():
            markdown_string += f"| {key} | {value['temporal behavior']} | {value['description']} |\n"
    # Initialize Markdown table header
    return markdown_string

def get_sensor_failure_association():
    import json
    with open('./ijcairesults/converted_dict.json') as f:
        data = json.load(f) 

    markdown_string = "#### Sensor-Failure Association\n\n"
    markdown_string += "| Component | Failure Mode | Sensor | Temporal Behavior |\n| --- | --- | --- | --- |\n"

    # Iterate over dictionary items and format as Markdown table rows
    for Component, cvalue in data.items():
        for Failures, fvalue in cvalue.items():
            for Sensors, svalue in fvalue.items():
                markdown_string += f"| {Component} | {Failures} | {Sensors} | {svalue['temporal behavior']} |\n"

    return markdown_string


def collect_question_answer_messages(assetname='windturbinegearbox',model='mixtral'):
    """_summary_
    """
    global question, ds_answer, sme_answer, all_question

    import pandas as pd
    # questions,answers,round
    df1 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_ds_bank_{assetname}_{model}.csv')
    df2 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_sme_bank_{assetname}_{model}.csv')
    merged_df = pd.merge(df1, df2, on='questions', how='outer', suffixes=('_round1', '_round2'))
    merged_df = merged_df.sort_values(by='round_round1')
    merged_df.fillna('', inplace=True)
    question = merged_df['questions'].to_list()
    ds_answer = merged_df['answers_round1'].to_list()
    sme_answer = merged_df['answers_round2'].to_list()

    df3 = pd.read_csv(f'./results/{assetname}/genai_questions_bank_{assetname}_{model}.csv')
    merged_df3 = df3.sort_values(by='round')
    all_question = merged_df3['questions'].to_list()

def main():
    st.set_page_config(page_title="Anomaly Detection System")
    font_size = 24
    st.markdown(f"<h1 style='font-size:{font_size}px;'>CASY: Consumable Anomaly Detection System</h1>", unsafe_allow_html=True)
    st.markdown(
        """<style>.block-container{max-width: 66rem !important;}</style>""",
        unsafe_allow_html=True,
    )
    disclaimer = """
    <span style="color: gray; font-size: 14px;">
        Disclaimer: This is a sample application and the information provided is for demonstration purposes only.
    </span>
    """
    st.markdown(disclaimer, unsafe_allow_html=True)
    with st.sidebar:

        st.markdown("## CASY ")
        st.markdown("""
            This is a sample app that demonstrates Mixture of Experts technology for a domain specific information generation.
        """)

        with st.form(key="my_form"):
            st.header("Input Configuration")
            selected_asset = st.selectbox(
                "Select an Asset Class",
                assets,
                key="editable_dropdown_asset",
                format_func=lambda x: x,
            )
            selected_model = st.selectbox(
                "Select LLM",
                ["Mixtral", "LLAMA"],
                key="editable_dropdown_model",
                format_func=lambda x: x,
            )
            st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

            with st.container():
                st.header("Output Configuration")
                num_qa = st.radio("How many Sensor-Failure Assoication Examples?", [5, "Skip", "All"])
                num_questions = st.radio("How many Temporal Behaviour Code Generation Examples?", [5, "Skip", "All"])

            st.markdown(
                    "<style>div[data-testid='stFormSubmitButton'] {display: flex; justify-content: center;}</style>",
                    unsafe_allow_html=True,
                )

            submit_button = st.form_submit_button(label="Run")

    # Use a container to store chat messages
    if submit_button and selected_asset and selected_model:
        st.empty()

        # Send the user input to FastAPI for processing
        # st.session_state["past"].append(selected_item)
        # st.session_state["generated"].append(selected_item)

        # Call FastAPI to submit the user input
        # submit_request(selected_asset)

        # For now, simulating a response
        with st.chat_message("ai", avatar="#️⃣"):
            display_message("""Welcome to CASY! I am a responsible and automated (multi-agent) AI system designed to assist you in generating domain specific sensor-failure association for a given asset. Additionally, I have the skill to generate python code for given temporal behaviour of time series data. \n \n My goal is to auto-generate a sensor-failure association and pattern detection library based on a selected asset class. In this demonstration, I will provide a preview of a few examples of my work. Let's begin!""")
            time.sleep(7)

            with st.spinner("Inviting team members (Data Scientist 👨‍🔬, Subject Matter Expert 🧑‍🏭, etc)..."):
                time.sleep(5)
                st.success("Members invited!")

        col1, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed
        with col2:
            with st.chat_message("user", avatar='👨‍🔬'):
                st.write("I am Data Scientist Expert.")

                with st.spinner(" I am getting ready ..."):
                    time.sleep(5)
                    st.success("I am ready!")

        with col2:
            with st.chat_message("user", avatar='🧑‍🏭'):
                st.write("I am Subject Matter Expert.")

                with st.spinner(" I am getting ready ..."):
                    time.sleep(2)
                    st.success("I am ready!")

        with st.spinner("AI is processing..."):
            assetname = selected_asset.lower().replace(' ','')
            modelname = selected_model.lower().replace(' ','')
            collect_system_messages(assetname, modelname)
            collect_question_answer_messages(assetname, modelname)
            time.sleep(2)

        # display system messages
        with st.chat_message("ai", avatar="#️⃣"):
            display_message("Let me set the agenda! \n \n")
            display_message(system_prompt + '. First, generate background document for the given asset.')
            time.sleep(2)

        _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

        with col2:
            with st.chat_message("user", avatar='🧑‍🏭'):
                display_message(sme_prompt)

        time.sleep(2)

        with col2:
            with st.chat_message("user", avatar='👨‍🔬'):
                display_message(ds_promt)

        sensor_failure_tags()
        total_ans = len(smePrompt)
        # display system messages
        with st.chat_message("ai", avatar="#️⃣"):
            display_message('I will ask few question to ' + '🧑‍🏭' + '! \n \n')
            time.sleep(2)

        for i in range(total_ans):

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('❓' + '\n' + smePrompt[i])

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            with col2:
                with st.chat_message("user", avatar='🧑‍🏭'):
                    if '{' in smeAnswer[i]:
                        display_message(get_markdown(smeAnswer[i],i))  
                    elif '1.' in smeAnswer[i]:
                        import re
                        output_string = re.sub(r'(\d+)', r'\n\n\1', smeAnswer[i])
                        display_message(output_string)  
                    else:
                        display_message(smeAnswer[i])
                    score = reduction_factor(smeAnswer[i]) 
                    display_chat_message_warning(score)

        for i in range(1):

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('Let us summarize the Sensor-Failure Association! \n \n')
                time.sleep(2)

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            with col2:
                markdowna = get_sensor_failure_association()
                with st.chat_message("user", avatar='🧑‍🏭'):
                    display_message(markdowna)


        pattern_template_library()
        with st.chat_message("ai", avatar="#️⃣"):
            display_message('I will ask code generation to ' + '👨‍🔬' + '! \n \n')
            time.sleep(2)
        total_ans = len(dmessage)
        for i in range(3):

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('❓' + '\n' + dmessage[i])

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            with col2:
                with st.chat_message("user", avatar='👨‍🔬'):
                    display_message("```python \n" + pcode_ans[i])

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('❓' + '\n' + 'Generate python code with single function that produce synthetic time series for above code.')

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            with col2:
                with st.chat_message("user", avatar='👨‍🔬'):
                    # Split the string at the second occurrence of "def"
                    cut_string = scode_ans[i].split("def", 2)[:2]
                    # Join the cut parts back into a string
                    cut_string = "".join(cut_string)
                    display_message("```python \n" + cut_string)

if __name__ == "__main__":
    main()

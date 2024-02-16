# app.py
import streamlit as st
import matplotlib.pyplot as plt
import time

system_prompt = ''
ds_promt = ''
sme_prompt = ''
sf_prompt = ''
re_prompt = ''
qe_prompt = ''

question = []
ds_answer = []
sme_answer = []
all_question = []
sf_answer = []
qe_answer = []
re_answer = []


def get_failure_location_level(data, key="Failure Location", level=1):
    for k, v in data.items():
        if k == key:
            return level
        elif isinstance(v, dict):
            nested_level = get_failure_location_level(v, key, level + 1)
            if nested_level:
                return nested_level
    return None

def traverse_tree(data):
    result = []
    if isinstance(data, dict):
        result.append({'name': data['name'], 'type': data['type'], 'description': data['description']})
        if 'components' in data.keys():
            for item in data['components']:
                result.extend(traverse_tree(item))
        elif 'Components' in data.keys():
            for item in data['components']:
                result.extend(traverse_tree(item))
        elif 'subcomponents' in data.keys():
            for item in data['subcomponents']:
                result.extend(traverse_tree(item))
        elif 'Subcomponents' in data.keys():
            for item in data['Subcomponents']:
                result.extend(traverse_tree(item))
        elif 'SubComponents' in data.keys():
            for item in data['SubComponents']:
                result.extend(traverse_tree(item))
        else:
            pass
    return result

def flatten_failure_data(data, level):
    flat_data = []
    if level == 4:
        for _, failures in data.items():
            for _, details in failures.items():
                for _, details1 in details.items():
                    flat_data.append(details1)
    elif level == 3:
        for _, failures in data.items():
            for _, details in failures.items():
                flat_data.append(details)
    elif level == 2:
        for _, details in failures.items():
            flat_data.append(details)
    return flat_data

assets = ["Electrical Submersible Pump", "Pumps", "Wind Turbine Gearbox", "Standby Generator", "Air Compressor", "Hydroelectric Power Turbine", "Electrical Transformer", "Induced Draft Fan", "Blast Furnace", "Electric Battery", "Substation Electrical Transformer","Water-Cooled Condenser","Industrial Robot", "Turbine Generator", "Industrial boiler", "Industrial Oven", "Industrial Furnace", "Centrifugal Compressor", "Hydraulic Press", "Steam Turbine"]
applications = ["FMEA Generation", "Anomaly Detection"]

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

def cross_check_file():
    pass

def display_component_list(assetname='electricalsubmersiblepump',model='granite'):
    import pandas as pd 
    import json
    filename = f'./results/{assetname}/component_list.json'
    with open(filename, "r") as f:
        stored_data = json.load(f)
    
    ddata = []
    # Create a Markdown table with rows
    markdown_str = "### Component Details\n"
    markdown_str += "| **Name** | **Type** | **Description**  |\n"
    markdown_str += "|----------------------|-------------------|---------------------|\n"
    for pdata in stored_data:
        tmp_data = traverse_tree(pdata)
        for data in tmp_data:
            ddata.append([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
            markdown_str += "| "
            markdown_str += " | ".join([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
            markdown_str += " |\n"
        
    df = pd.DataFrame(ddata, columns=['Name', 'Type', 'Description'])
    df.to_csv('component.csv',index=False)
    print(df)

    return markdown_str 

@st.cache_data
def display_finalized_component_list(assetname='electricalsubmersiblepump',model='granite'):
    import pandas as pd 
    filename = f'./results/{assetname}/finalized_component.csv'
    df = pd.read_csv(filename)
    
    # Create a Markdown table with rows
    markdown_str = "### Component Details\n"
    markdown_str += "| **Name** | **Description**  |\n"
    markdown_str += "|----------------------|---------------------|\n"
    for _, pdata in df.iterrows():
        markdown_str += "| "
        markdown_str += " | ".join([", ".join(map(str, value)) if isinstance(value, list) else str(value) for value in pdata.values])
        markdown_str += " |\n"
        
    return markdown_str, df

@st.cache_data
def display_final_failure_list(assetname='electricalsubmersiblepump',model='granite'):
    import pandas as pd 
    import json
    filename = f'./results/{assetname}/final_failures.csv'
    stored_data = pd.read_csv(filename)
    # Create a Markdown table with rows
    markdown_str = "### Failure Details\n"
    markdown_str += "| **Failure Location** | **Failure Mode** | **Failure Causes** | **Failure Effects** |\n"
    markdown_str += "|----------------------|-------------------|---------------------|---------------------|\n"
    for _, pdata in stored_data.iterrows():
        markdown_str += "| "
        markdown_str += " | ".join([str(value) if not isinstance(value, list) else ", ".join(value) for value in pdata.values])
        markdown_str += " |\n"
        
    return markdown_str, stored_data


def display_failure_list(assetname='electricalsubmersiblepump',model='granite'):
    import pandas as pd 
    import json
    filename = f'./results/{assetname}/failure_list.json'
    with open(filename, "r") as f:
        stored_data = json.load(f)
    # Create a Markdown table with rows
    fdata = []
    markdown_str = "### Failure Details\n"
    markdown_str += "| **Failure Location** | **Failure Mode** | **Failure Causes** | **Failure Effects** |\n"
    markdown_str += "|----------------------|-------------------|---------------------|---------------------|\n"
    for data in stored_data:
        lvl = get_failure_location_level(data)
        if lvl == 4 or lvl == 2 or lvl == 3:
            tmpdata = flatten_failure_data(data, lvl)
            for data in tmpdata:
                fdata.append([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
                markdown_str += "| "
                markdown_str += " | ".join([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
                markdown_str += " |\n"
        else:
            fdata.append([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
            markdown_str += "| "
            markdown_str += " | ".join([str(value) if not isinstance(value, list) else ", ".join(value) for value in data.values()])
            markdown_str += " |\n"
        
    df = pd.DataFrame(fdata, columns=['Failure Location', 'Failure Mode', 'Failure Causes', 'Failure Effects'])
    df.to_csv('failure.csv',index=False)
    return markdown_str 

def collect_system_messages(assetname='electricalsubmersiblepump',model='granite'):
    """_summary_
    """
    global system_prompt, sf_promt, sme_prompt, re_prompt, qe_prompt

    import pandas as pd 
    df = pd.read_csv(f'./results/{assetname}/genai_context_docs_{assetname}_{model}.csv')
    system_prompt = list(df['context_prompt'])[0]
    sf_promt = list(df['sf_context'])[0]
    sme_prompt = list(df['sme_context'])[0]
    qe_prompt = list(df['qe_context'])[0]
    re_prompt = list(df['re_context'])[0]
    qe_prompt =  list(df['qe_context'])[0]
    
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

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

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

def collect_question_answer_messages(assetname='electricalsubmersiblepump',model='granite'):
    """_summary_
    """
    global question, sf_answer, sme_answer, all_question, qe_question, re_question, qe_answer, re_answer

    import pandas as pd
    # questions,answers,round
    df1 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_sf_bank_{assetname}_{model}.csv')
    df2 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_sme_bank_{assetname}_{model}.csv')
    df3 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_qe_bank_{assetname}_{model}.csv')
    df4 = pd.read_csv(f'./results/{assetname}/genai_questions_answer_re_bank_{assetname}_{model}.csv')
    df1.columns = ['questions', 'answers_sf', 'round_sf']
    df2.columns = ['questions', 'answers_sme', 'round_sme']
    df3.columns = ['questions', 'answers_qe', 'round_qe']
    df4.columns = ['questions', 'answers_re', 'round_re']

    merged_df = pd.merge(df1, df2, on='questions', how='outer')
    merged_df = pd.merge(merged_df, df3, on='questions', how='outer')
    merged_df = pd.merge(merged_df, df4, on='questions', how='outer')

    merged_df = merged_df.sort_values(by='round_sme')
    merged_df = merged_df[merged_df['questions'].str.contains('failure|modes|components|critical|component|maintenance|faults|codes|reasons|degradation|mechanism')]
    merged_df.to_csv('testing_data.csv',index=False)
    merged_df.fillna('', inplace=True)
    question = merged_df['questions'].to_list()
    sf_answer = merged_df['answers_sf'].to_list()
    sme_answer = merged_df['answers_sme'].to_list()
    qe_answer = merged_df['answers_qe'].to_list()
    re_answer = merged_df['answers_re'].to_list()

    df3 = pd.read_csv(f'./results/{assetname}/genai_questions_bank_{assetname}_{model}.csv')
    merged_df3 = df3.sort_values(by='round')
    all_question = merged_df3['questions'].to_list()

def main():
    st.set_page_config(page_title="Question Generation AI Recipe")
    font_size = 24
    st.markdown(f"<h1 style='font-size:{font_size}px;'>Auto-Qx5: Automated Question Generation for Industrial Assets using Mixture of Agents</h1>", unsafe_allow_html=True)
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
    cross_check_file()
    with st.sidebar:
        st.markdown("## Auto-Qx5 System")
        st.markdown("""
            This is a sample app that demonstrates Mixture of Experts technology for a domain specific question generation.
        """)

        with st.form(key="my_form"):
            st.header("Input Configuration")
            selected_asset = st.selectbox(
                "Select an Asset Class",
                assets,
                key="editable_dropdown_asset",
                format_func=lambda x: x,
            )
            selected_application = st.selectbox(
                "Select an Application",
                applications,
                key="editable_dropdown_application",
                format_func=lambda x: x,
            )
            selected_model = st.selectbox(
                "Select LLM",
                ["Granite (Labrador)", "Mixtral", "LLAMA"],
                key="editable_dropdown_model",
                format_func=lambda x: x,
            )
            selected_model = selected_model.split(' ')[0]
            st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

            with st.container():
                st.header("Output Configuration")
                num_qa = st.radio("How many Question-Answer pairs?", [5, "Skip", "All"])
                num_questions = st.radio("How many additional Questions?", [5, "Skip", "All"])

            st.markdown(
                    "<style>div[data-testid='stFormSubmitButton'] {display: flex; justify-content: center;}</style>",
                    unsafe_allow_html=True,
                )

            submit_button = st.form_submit_button(label="Run")

    # Use a container to store chat messages
    # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    if submit_button and selected_asset and selected_model:
        st.empty()
        with st.chat_message("ai", avatar="#️⃣"):
            display_message("""Welcome to Auto-Qx5 System! I am a responsible and automated (multi-agent) AI system designed to assist you in generating domain specific questions. Additionally, I have the skill to select the right persona who can provide answers to the questions. \n \n My goal is to auto-generate a list of questions based on a selected asset class. In this demonstration, I will provide a preview of a few examples of questions and answers. Let's begin!""")
            time.sleep(7)
            if selected_application == 'FMEA Generation':
                with st.spinner("Inviting team members (FMEA Facilitator 👨‍🔬, Subject Matter Expert 🧑‍🏭, Quality Enginner 👨‍💼, Reliability Enginner 👨‍🔧)..."):
                    time.sleep(5)
                    st.success("Members invited!")
            elif selected_application == 'Anomaly Detection':
                with st.spinner("Inviting team members (Data Scientist 👨‍💻, Subject Matter Expert 🧑‍🏭, etc)..."):
                    time.sleep(5)
                    st.success("Members invited!")
            else:
                pass

        _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed
        with col2:
            if selected_application == 'FMEA Generation':
                with st.chat_message("user", avatar='👨‍🔬'):
                    st.write("I am FMEA Facilitator.")

                    with st.spinner(" I am getting ready ..."):
                        time.sleep(5)
                        st.success("I am ready!")

                with st.chat_message("user", avatar='🧑‍🏭'):
                    st.write("I am Subject Matter Expert.")

                    with st.spinner(" I am getting ready ..."):
                        time.sleep(2)
                        st.success("I am ready!")

                with st.chat_message("user", avatar='👨‍💼'):
                    st.write("I am Quality Engineer.")

                    with st.spinner(" I am getting ready ..."):
                        time.sleep(2)
                        st.success("I am ready!")

                with st.chat_message("user", avatar='👨‍🔧'):
                    st.write("I am Reliability Engineer.")

                    with st.spinner(" I am getting ready ..."):
                        time.sleep(2)
                        st.success("I am ready!")
 
            elif selected_application == 'Anomaly Detection':
                with st.chat_message("user", avatar='👨‍💻'):
                    st.write("I am Data Scientist Expert.")

                    with st.spinner(" I am getting ready ..."):
                        time.sleep(5)
                        st.success("I am ready!")

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

            if selected_application == 'FMEA Generation':

                skip = False
                if not skip:
                    with st.chat_message("user", avatar='🧑‍🏭'):
                        display_message(sme_prompt)

                    time.sleep(2)

                    with st.chat_message("user", avatar='👨‍💼'):
                        display_message(qe_prompt)

                    time.sleep(2)

                    with st.chat_message("user", avatar='👨‍🔧'):
                        display_message(re_prompt)

            else:
                with st.chat_message("user", avatar='🧑‍🏭'):
                    display_message(sme_prompt)

                time.sleep(2)

                with st.chat_message("user", avatar='👨‍🔬'):
                    display_message(ds_promt)

        # display system messages
        with st.chat_message("ai", avatar="#️⃣"):
            display_message(f"I have generated a total of {len(all_question)} questions. Of these, I have provided answers for {len(question)} questions. Based on the output configuration, I will now provide the generated questions and answers for your review. Please carefully review them.")
            display_message("\n")
            _, col2, _ = st.columns([0.25,0.50,0.25])  # Adjust column widths as needed
            with col2:
                plot_results(total_questions = len(all_question), answered_questions = len(question))

        # display tables
        total_ans = num_qa
        if num_qa == 'All':
            total_ans = len(question)
        elif num_qa == 'Skip':
            total_ans = 0

        for i in range(total_ans):

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('❓' + '\n' + question[i])

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            if len(sme_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='🧑‍🏭'):
                        display_message(sme_answer[i])
                        score = reduction_factor(sme_answer[i]) 
                        display_chat_message_warning(score)

            if len(sf_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='👨‍🔬'):
                        display_message(sf_answer[i])
                        score = reduction_factor(sf_answer[i]) 
                        display_chat_message_warning(score)

            if len(re_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='👨‍💼'):
                        display_message(re_answer[i])
                        score = reduction_factor(re_answer[i]) 
                        display_chat_message_warning(score)

            if len(qe_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='👨‍🔧'):
                        display_message(qe_answer[i])
                        score = reduction_factor(qe_answer[i]) 
                        display_chat_message_warning(score)

        # display summary
        with st.chat_message("ai", avatar="#️⃣"):
            display_message("Let us proceed to generate FMEA documentation! \n")

        _, col21 = st.columns([0.25,4.75])  # Adjust column widths as needed
        with col21:
            c_ans, my_large_df = display_finalized_component_list()
            with st.chat_message("user", avatar='👨‍🔬'):
                display_message("First, Let us look at Component, Subcomponent and assemblies of selected asset \n")
                display_message(c_ans)
                comp_csv = convert_df(my_large_df)
                display_message('\n')
                st.download_button("Download " + "⬇️", data=comp_csv,file_name=f'./results/{assetname}/finalized_component.csv',mime='text/csv')
                
        _, col21 = st.columns([0.25,4.75])  # Adjust column widths as needed
        with col21:
            c_ans, f_stored_data = display_final_failure_list()
            with st.chat_message("user", avatar='👨‍🔬'):
                display_message("Now, Let us look at Failure Modes of selected asset \n")
                display_message(c_ans)
                failure_csv = convert_df(f_stored_data)
                display_message('\n')
                st.download_button("Download " + "⬇️", data=failure_csv,file_name=f'./results/{assetname}/finalized_failure.csv',mime='text/csv')

        # display questions
        total_ans = num_questions
        if num_questions == 'All':
            total_ans = len(all_question)
        elif num_questions == 'Skip':
            total_ans = 0
        if total_ans > 0:
            with st.chat_message("ai", avatar="#️⃣"):
                display_message("System has generated following additional questions, which are yet pending for anaswer generation.")
        _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed
        for item in range(len(all_question)):
            if all_question[item] not in question:
                total_ans = total_ans - 1
                time.sleep(2)
                with col2:
                    with st.chat_message("ai", avatar="❓"):
                        display_message(all_question[item])
            if total_ans == 0:
                break

if __name__ == "__main__":
    main()

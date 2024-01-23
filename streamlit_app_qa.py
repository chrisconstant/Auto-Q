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
    st.set_page_config(page_title="Question Generation AI Recipe")
    st.title("Auto-Qx5 : Automated Multi-Agent System")
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

        st.markdown("## Auto-Qx5 System")
        st.markdown("""
            This is a sample app that demonstrates Mixture of Experts technology for a domain specific question generation.
        """)

        with st.form(key="my_form"):
            st.header("Input Configuration")
            selected_asset = st.selectbox(
                "Select an Asset Class",
                ["Wind Turbine Gearbox", "Standby Generator", "Chiller"],
                key="editable_dropdown_asset",
                format_func=lambda x: x,
            )
            selected_model = st.selectbox(
                "Select LLM",
                ["LAMMA", "Mixtral", "Granite"],
                key="editable_dropdown_model",
                format_func=lambda x: x,
            )
            st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

            with st.container():
                st.header("Output Configuration")
                num_qa = st.radio("How many Question-Answer pairs?", [5, 10, "All"])
                num_questions = st.radio("How many additional Questions?", [5, 10, "All"])

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
            display_message("""Welcome to Auto-Qx5 System! I am a responsible and automated AI system designed to help you generate questions. I also have a skil to select the right persona who can answer the question. \n \n In this demo, I will provide a preview of a few examples of questions and answers. Let's get started!""")
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

        # display system messages
        with st.chat_message("ai", avatar="#️⃣"):
            display_message(f"I have generated total {len(all_question)} questions. Out of these, I have generated answer for {len(question)} questions. Based on output configuration, I will now provide generated questions and answers. Please review it with carefully!")
            display_message("\n")
            _, col2, _ = st.columns([0.25,0.50,0.25])  # Adjust column widths as needed
            with col2:
                plot_results()

        # display tables
        total_ans = num_qa
        if num_qa == 'All':
            total_ans = len(question)
        for i in range(total_ans):

            with st.chat_message("ai", avatar="#️⃣"):
                display_message('❓' + '\n' + question[i])

            _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed

            if len(sme_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='🧑‍🏭'):
                        display_message(sme_answer[i])

            if len(ds_answer[i]) > 0:
                with col2:
                    with st.chat_message("user", avatar='👨‍🔬'):
                        display_message(ds_answer[i])
                        
        # display questions
        with st.chat_message("ai", avatar="#️⃣"):
            display_message("System has generated following additional questions, which are yet pending for anaswer generation.")
        _, col2 = st.columns([0.25,4.75])  # Adjust column widths as needed
        total_ans = num_questions
        if num_questions == 'All':
            total_ans = len(all_question)
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

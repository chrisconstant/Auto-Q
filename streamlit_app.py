# app.py
import streamlit as st
from streamlit_chat import message
import requests
import asyncio


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "update_task" not in st.session_state:
    st.session_state["update_task"] = False

def submit_request(user_input):
    url = "http://localhost:8001/submit-request/"
    payload = {"user_input": user_input}
    response = requests.post(url, json=payload)
    return response.json()


def get_response_async():
    url = "http://localhost:8001/get-response/"
    response = requests.get(url)
    return response.json()

def update_responses():
    response = get_response_async()
    st.session_state["past"].append(response["response"])
    st.session_state["generated"].append(response["response"])

def main():
    st.set_page_config(page_title="AI Recipe")
    st.title("Zero-Shot Multi-Agent AI System")

    if len(st.session_state["generated"]) == 0:
        st.session_state["past"].append("I am Data Scientist Agent")
        st.session_state["generated"].append("I am Subject Matter Expert Agent")

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            selected_item = st.selectbox(
                "Select an Asset Class",
                ["Wind Turbine", "Standby Generator", "Chiller"],
                key="editable_dropdown",
                format_func=lambda x: x,
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and selected_item:
            # Send the user input to FastAPI for processing
            #st.session_state["past"].append(selected_item)
            #st.session_state["generated"].append(selected_item)

            # Call FastAPI to submit the user input
            submit_request(selected_item)

            # For now, simulating a response
            st.session_state["past"].append("I am getting ready ...")
            st.session_state["generated"].append("I am also getting ready...")
            st.session_state["update_task"] = True

    if st.session_state["update_task"]:
        if st.session_state["generated"]:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                    message(st.session_state["generated"][i], key=str(i))
            
            st.session_state["update_task"] = False

    if st.session_state["update_task"]:
        update_responses()

if __name__ == "__main__":
    main()

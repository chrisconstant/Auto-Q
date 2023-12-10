# main.py
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import time
import asyncio

app = FastAPI()

# Shared data between clients
shared_data = {"responses": [], "response_queues": []}

async def generate_response(user_input: str, background_tasks: BackgroundTasks):
    for i in range(5):  # Simulating incremental responses
        await asyncio.sleep(2)  # Simulating some work asynchronously
        response = f"Response {i + 1} to '{user_input}'"
        shared_data["responses"].append(response)
        background_tasks.add_task(push_response, response)

async def push_response(response: str):
    for queue in shared_data["response_queues"]:
        await queue.put(response)


def get_response_queue():
    response_queue = asyncio.Queue()
    shared_data["response_queues"].append(response_queue)
    return response_queue

@app.post("/submit-request/")
async def submit_request(user_input: str, background_tasks: BackgroundTasks):
    response_queue = get_response_queue()
    background_tasks.add_task(generate_response, user_input, background_tasks)
    
    return JSONResponse(content={"message": "Request submitted"})


@app.get("/get-response/")
async def get_response(response_queue: asyncio.Queue = Depends(get_response_queue)):
    response = await response_queue.get()
    return JSONResponse(content={"response": response})

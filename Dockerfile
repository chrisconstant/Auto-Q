# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install streamlit==1.29.0 streamlit-chat==0.1.1 ibm-generative-ai[langchain]
RUN pip install fastapi uvicorn
RUN pip install .

# Make port 8501 available to the world outside this container
EXPOSE 8501 8001

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["sh", "-c", "streamlit run streamlit_app.py & uvicorn fastapi_app:app --host 0.0.0.0 --port 8001 --reload"]

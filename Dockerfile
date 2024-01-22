# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.9-slim 

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install any needed packages specified in requirements.txt
RUN pip install streamlit==1.29.0 streamlit-chat==0.1.1
# RUN pip install .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "streamlit_app_qa.py", "--server.port=8501", "--server.address=0.0.0.0"]

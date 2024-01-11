# AutoQA Survey App

This is a Flask web application for conducting expert evaluations in the field of AutoQA. It allows experts to provide their opinions on various questions related to wind turbine gearbox analysis.

## Running Locally

1. Clone the repository:

   ```bash
   git clone -b simplified_chat https://github.ibm.com/pateldha/AutoQA.git
   ```

2. Navigate to the project directory:

    ```bash
    cd AutoQA/survey 
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask app with Gunicorn (suitable for production):
    ```bash
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
    ```

The app will be accessible at http://localhost:5000.


## Building Docker Image

1. Make sure Docker is installed on your machine.

2. Build and push the Docker image:

    ```bash
    docker build -t us.icr.io/aimodelfactory/autoqa-surveyapp:1.0.0 .
    docker push us.icr.io/aimodelfactory/autoqa-surveyapp:1.0.0
    ```

Replace `us.icr.io/aimodelfactory/autoqa-surveyapp:1.0.0` with your desired Docker image name and tag.


## Deploying on OpenShift

1. Make sure the OpenShift CLI (oc) is installed on your machine.

2. Log in to your OpenShift cluster.

3. Create a namespace on your Openshift Cluster:

    ```bash
    oc create ns autoqa
    ```

4. Apply the deployment and service configuration:

    ```bash
    oc apply -f deployment.yaml -n autoqa
    ```

Congratulations! Your Flask app is now running on OpenShift with Gunicorn as the WSGI server. You can access it here - https://autoqa-surveyapp.modelfactory-9ca4d14d48413d18ce61b80811ba4308-0000.us-south.containers.appdomain.cloud/
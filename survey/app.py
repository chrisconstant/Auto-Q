from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, distinct
import ast

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_responses.db'
db = SQLAlchemy(app)

class UserResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    evaluation_number = db.Column(db.String(255)) 
    question_number = db.Column(db.String(255))
    selected_option = db.Column(db.String(255))

class TopicSurveyResponse(db.Model):
    id = db.Column(db.String(255), primary_key=True)
    humanness_group = db.Column(db.String(1))
    topic_coverage_group = db.Column(db.String(1))
    engagement_group = db.Column(db.String(1))
    novelty_group = db.Column(db.String(1))

with app.app_context():
    db.create_all()

def read_questions_from_csv(file_path, separator=','):
    questions = []
    import pandas as pd
    df = pd.read_csv(file_path)
    df['Question'] = df['Question'].apply(lambda x: f"{x} \n\n Whom do you think this question is appropriate for?")
    question = list(df['Question'])
    real_persona = list(df['Real Persona'])
    choices = df[['Persona Option 1','Persona Option 2','Persona Option 3','Persona Option 4']].to_numpy().tolist()
    for i in range(len(question)):
        questions.append({'question': question[i], 'choices': choices[i], 'real_persona': real_persona[i]})
    return questions

@app.route('/redirect_survey', methods=['POST'])
def redirect_survey():
    selected_option = request.form.get('survey_option')

    if selected_option == 'option1':
        return redirect(url_for('questionclassifier'))  # Redirect to webpage1
    elif selected_option == 'option2':
        return redirect(url_for('topicsurvey'))  # Redirect to webpage2
    else:
        # Handle other cases or errors
        return redirect(url_for('index'))

def read_topics_from_csv(file_path):
    """_summary_

    :param file_path: _description_
    :type file_path: _type_
    :param separator: _description_, defaults to ','
    :type separator: str, optional
    :return: _description_
    :rtype: _type_
    """
    questions = []
    import pandas as pd
    df = pd.read_csv(file_path)
    df['Name'] = df['Name'].apply(lambda x: x.split("_", 1)[1].replace("_", ", "))
    df['Representative_Docs'] = df['Representative_Docs'].apply(lambda x: ast.literal_eval(x))
    rep = list(df['Name'])
    freq = list(df['Count'])

    Representation = list(df['Representation'])
    Representative_Docs = list(df['Representative_Docs'])
    for i in range(len(rep)):
        questions.append({'number': 'Topic Id ' + str(i), 'rep': rep[i], 'freq': freq[i], 'repr': Representation[i], 'sentence_list': Representative_Docs[i]})
    return questions

# Define the CSV file path and separator
csv_file_path = 'questions.csv'
csv_separator = ','  # Change this to your desired separator

# Read questions from the CSV file with the specified separator
questions = read_questions_from_csv(csv_file_path, separator=csv_separator)

# Add question numbers to the questions list
questions_with_numbers = [{'number': f'Q{i}', **question} for i, question in enumerate(questions, 1)]

# topic
# Define the CSV file path and separator
csv_file_path_1 = 'top10_recipe.csv'
topics = read_topics_from_csv(csv_file_path_1)

# Main route to display the survey
@app.route('/topicsurvey')
def topicsurvey():
    return render_template('topic_survey.html', topics=topics)

# Main route to display the survey
@app.route('/questionclassifier')
def questionclassifier():
    return render_template('survey.html', questions=questions_with_numbers)

# Main route to display the survey
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    user_responses = {key: request.form[key] for key in request.form}

    # save
    save_responses_to_database(user_responses)

    # Get the real persona for the submitted questions
    real_personas = [question['real_persona'] for question in questions]
    
    return render_template('thank_you.html', responses=user_responses, questions=questions_with_numbers, real_personas=real_personas)

@app.route('/submit_topic', methods=['POST'])
def submit_topic():
    user_responses = {key: request.form[key] for key in request.form}

    # save
    save_topic_responses_to_database(user_responses)
    
    return render_template('thank_you_simple.html', responses=user_responses)

# Route to handle question-wise summary
@app.route('/summary')
def summary():
    # Use SQLAlchemy's func.count to get the count of each option for each question
    summary_data = db.session.query(UserResponse.question_number, UserResponse.selected_option, func.count()).group_by(UserResponse.question_number, UserResponse.selected_option).all()

  # Calculate the total number of questions
    total_questions = db.session.query(func.count(distinct(UserResponse.question_number))).scalar()

    # Calculate the total number of responses
    total_responses = db.session.query(func.count(UserResponse.id)).scalar()

    return render_template('summary.html', summary_data=summary_data, total_questions=total_questions, total_responses=total_responses)


# Route to clear the database
@app.route('/cleardb')
def clear_database():
    try:
        # Delete all records from the UserResponse table
        db.session.query(UserResponse).delete()

        # Commit the changes
        db.session.commit()

        message = "Database cleared successfully."
    except Exception as e:
        # Handle exceptions if any
        message = f"Error clearing database: {str(e)}"

    return render_template('admin_action.html', message=message)
    
def save_responses_to_database(user_responses):
    print (user_responses)
    import uuid
    random_uuid = str(uuid.uuid4())

    with app.app_context():
        # Extract question number and selected option from user_responses
        for question_number, selected_option in user_responses.items():
            if 'action' not in question_number:
                new_response = UserResponse(evaluation_number=random_uuid, question_number=question_number, selected_option=selected_option)
                db.session.add(new_response)

        db.session.commit()

def save_topic_responses_to_database(user_responses):
    print (user_responses)
    import uuid
    random_uuid = str(uuid.uuid4())

    with app.app_context():

        new_response = TopicSurveyResponse(
                id=random_uuid,
                humanness_group=user_responses.get('humanness_group'),
                topic_coverage_group=user_responses.get('topic_coverage_group'),
                engagement_group=user_responses.get('engagement_group'),
                novelty_group=user_responses.get('novelty_group')
            )
        db.session.add(new_response)
        db.session.commit()


if __name__ == '__main__':
    app.run(debug=True)


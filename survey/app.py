from flask import Flask, render_template, request

app = Flask(__name__)

def read_questions_from_csv(file_path, separator=','):
    questions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line (header)
        for line in lines:
            parts = line.strip().split(separator)
            question = f"\"{parts[0]}\"\n\n Whom do you think this question is appropriate for?"
            # Exclude the second column (real answer) from choices
            choices = parts[2:]
            real_persona = parts[1]
            questions.append({'question': question, 'choices': choices, 'real_persona': real_persona})
    return questions

# Define the CSV file path and separator
csv_file_path = 'questions.csv'
csv_separator = ';'  # Change this to your desired separator

# Read questions from the CSV file with the specified separator
questions = read_questions_from_csv(csv_file_path, separator=csv_separator)

# Add question numbers to the questions list
questions_with_numbers = [{'number': f'Q{i}', **question} for i, question in enumerate(questions, 1)]

# Main route to display the survey
@app.route('/')
def index():
    return render_template('survey.html', questions=questions_with_numbers)

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    user_responses = {key: request.form[key] for key in request.form}
    
    # Get the real persona for the submitted questions
    real_personas = [question['real_persona'] for question in questions]
    
    return render_template('thank_you.html', responses=user_responses, questions=questions_with_numbers, real_personas=real_personas)

if __name__ == '__main__':
    app.run(debug=True)

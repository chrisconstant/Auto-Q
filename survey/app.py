from flask import Flask, render_template, request

app = Flask(__name__)

# Function to read questions and answers from CSV with a custom separator
def read_questions_from_csv(file_path, separator=','):
    questions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(separator)
            question = f"\"{parts[0]}\"\n\nWhom do you think this question is appropriate for?"
            choices = parts[1:]
            questions.append({'question': question, 'choices': choices})
    return questions



# Sample CSV file structure:
# Question 1;Choice A;Choice B;Choice C;Choice D
# Question 2;Choice X;Choice Y;Choice Z

# Define the CSV file path and separator
csv_file_path = 'questions.csv'
csv_separator = ';'  # Change this to your desired separator

# Read questions from the CSV file with the specified separator
questions = read_questions_from_csv(csv_file_path, separator=csv_separator)

# Add question numbers to the questions list
questions_with_numbers = [{'number': f'Q{i}', **question} for i, question in enumerate(questions, 1)]

# Main route to display the questionnaire
@app.route('/')
def index():
    return render_template('questionnaire.html', questions=questions_with_numbers)

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    user_responses = {key: request.form[key] for key in request.form}
    # Process user responses as needed
    return render_template('thank_you.html', responses=user_responses, questions=questions_with_numbers)

if __name__ == '__main__':
    app.run(debug=True)

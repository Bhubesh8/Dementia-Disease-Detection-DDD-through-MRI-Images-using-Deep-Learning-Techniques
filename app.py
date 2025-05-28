# from flask import Flask, render_template, request, redirect, url_for, session
# from datetime import date
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
 

# app = Flask(__name__)
# app.secret_key = 'Bhubi@29'  # Required for session management
# present_date = date.today()
# formatted_date = present_date.strftime("%d-%m-%Y")
# print(formatted_date)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# model = load_model("Dementia_Model_binary.h5")
# class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']


# questions = [
#     "Study the three word bellow take your time but do not write them down. In few minutes you will be asked.\n"
#     " Pomegranate, Alice, Blue",
#     "What is the date?(Format in DD-MM-YYYY)",
#     "Can you remember this address : '29/A Whites Road, Chennai' and recall it after few minutes?",
#     "Without paper or devices calculate '13*4'.",
#     "Take a look at the picture in few minutes you will be asked to recall it.",
#     "If you have 100rs and you go to store and buy a dozen chocolates for each it cost 3rs and strawberry for 40rs. How much did you spend and how much left?(Just mention numbers with comma)",
#     "Can you name the animal from the image below?",
#     "Take the pen and paper draw a clock showing the time as quater of eleven and mention answer below as 'HH:MM' format",
#     "What was the three words which was asked first?",
#     "What is the address which was asked before?",
#     "What is the name of the image which was asked before?",
#     "Till now how many different types of fruits were mentioned on this test?",
#     "what comes next in the pattern?"
# ]
# answers = [
#     'ok',
#     formatted_date,
#     'ok',
#     '52',
#     'ok',
#     '76,34',
#     'tiger',
#     '11:15',
#     'pomegranate,alice,blue',
#     '29/a whites road,chennai',
#     'mango',
#     '3',
#     'a'
# ]

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_tensor = image.img_to_array(img)
#     img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(img_tensor)
#     img_tensor = np.expand_dims(img_tensor, axis=0)

#     predictions = model.predict(img_tensor)
#     predicted_index = np.argmax(predictions)
#     confidence = float(np.max(predictions)) * 100

#     return class_labels[predicted_index], round(confidence, 2)


# @app.route('/')
# def start_quiz():
#     session['responses'] = []  # Reset responses at the beginning
#     session['question_index'] = 0  # Start from the first question
#     session['score'] = 0
#     return redirect(url_for('question'))

# @app.route('/question', methods=['GET', 'POST'])
# def question():
#     question_index = session.get('question_index', 0)
#     responses = session.get('responses', [])
#     score = session.get('score', 0)

#     if question_index >= len(questions):
#         return redirect(url_for('results'))

#     if request.method == 'POST':
#         response = request.form.get('response', '').strip().lower()
#         responses.append(response)

#         # Check answer only if it's not marked 'ok' in the answers list
#         correct_answer = answers[question_index].strip().lower()
#         if correct_answer != 'ok' and response == correct_answer:
#             score += 1

#         session['responses'] = responses
#         session['question_index'] = question_index + 1
#         session['score'] = score

#         return redirect(url_for('question'))

#     return render_template('quiz.html', question=questions[question_index], index=question_index + 1)

# @app.route('/results')
# def results():
#     responses = session.get('responses', [])
#     score = session.get('score', 0)
#     return render_template('results.html', responses=responses, score=score)

# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def index():
#     return render_template('index.html')
# def predict():
#     if 'image' not in request.files:
#         return "No file part"

#     file = request.files['image']
#     if file.filename == '':
#         return "No selected file"

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     prediction, confidence = predict_image(filepath)
#     return render_template('pred_result.html', prediction=prediction, confidence=confidence)

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)    


from flask import Flask, render_template, request, redirect, url_for, session
from datetime import date
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import uuid


app = Flask(__name__)
app.secret_key = 'Bhubi@29'
app.config['UPLOAD_FOLDER'] = 'uploads'
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

present_date = date.today()
formatted_date = present_date.strftime("%d-%m-%Y")

questions = [
    "Study the three word bellow take your time but do not write them down. In few minutes you will be asked just type 'ok'.\n Orange, Alice, Blue",
    "What is the date?(Format in DD-MM-YYYY)",
    "Can you remember this address : '29/A Whites Road, Chennai' and recall it after few minutes? Just type 'ok'",
    "Without paper or devices calculate '13*4'.",
    "Take a look at the picture in few minutes you will be asked to recall it. Just type 'ok'",
    "If you have 100rs and you go to store and buy a dozen chocolates for each it cost 3rs and strawberry for 40rs. How much did you spend and how much left?(Just mention numbers with comma)",
    "Can you name the animal from the image below?",
    "Take the pen and paper draw a clock showing the time as quater of eleven and mention answer below as 'HH:MM' format",
    "What was the three words which was asked first?",
    "What is the address which was asked before?",
    "What is the name of the image which was asked before?",
    "Till now how many different types of fruits were mentioned on this test?",
    "what comes next in the pattern?"
]

answers = [
    'ok',
    formatted_date,
    'ok',
    '52',
    'ok',
    '76,34',
    'tiger',
    '11:15',
    'orange, alice, blue',
    '29/a whites road, chennai',
    'mango',
    '3',
    'b'
]

def predict_image(img_path):
    model = load_model("Dementia_Model_binary.h5")
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    predictions = model.predict(img_tensor)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    return class_labels[predicted_index], round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_quiz')
def start_quiz():
    session['responses'] = []
    session['question_index'] = 0
    session['score'] = 0
    return redirect(url_for('question'))

@app.route('/question', methods=['GET', 'POST'])
def question():
    question_index = session.get('question_index', 0)
    responses = session.get('responses', [])
    score = session.get('score', 0)

    if question_index >= len(questions):
        return redirect(url_for('results'))

    if request.method == 'POST':
        response = request.form.get('response', '').strip().lower()
        responses.append(response)

        correct_answer = answers[question_index].strip().lower()
        if correct_answer != 'ok' and response == correct_answer:
            score += 1

        session['responses'] = responses
        session['question_index'] = question_index + 1
        session['score'] = score

        return redirect(url_for('question'))

    return render_template('quiz.html', question=questions[question_index], index=question_index + 1)

@app.route('/results')
def results():
    responses = session.get('responses', [])
    score = session.get('score', 0)
    return render_template('results.html', responses=responses, score=score)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return "No file part"

#     file = request.files['image']
#     if file.filename == '':
#         return "No selected file"

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     prediction, confidence = predict_image(filepath)
#     return render_template('pred_results.html', prediction=prediction, confidence=confidence)

@app.route('/upload')
def upload_page():
    return render_template('pred_result.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return "No file part"

#     file = request.files['image']
#     if file.filename == '':
#         return "No selected file"

#     # Ensure upload folder exists
#     upload_folder = os.path.join('static', 'uploads')
#     if not os.path.exists(upload_folder):
#         os.makedirs(upload_folder)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(upload_folder, filename)
#     file.save(filepath)

#     prediction, confidence = predict_image(filepath)

#     return render_template('pr_2.html', prediction=prediction, confidence=confidence, image_path=filepath)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('pred_result.html')

    file = request.files['image']
    if not file:
        return "No file selected"

    # Create upload directory if it doesn't exist
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    # Save the file with a unique name
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Run your prediction function here (you can replace this line)
    prediction, confidence = predict_image(filepath)

    # Path to show on web (relative to /static/)
    image_url = url_for('static', filename='uploads/' + filename)

    return render_template('pr_2.html', prediction=prediction, confidence=confidence, image_url=image_url)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

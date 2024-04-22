from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from naiveBayes import multinomial_naive_bayes, bernoulli_naive_bayes, gaussian_naive_bayes
from pso_NB import pso_multinomial_naive_bayes, pso_bernoulli_naive_bayes, pso_gaussian_naive_bayes

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return render_template('classification.html', file_path=file_path)
        else:
            return render_template('index.html', error='Invalid file type')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        file_path = request.form['file_path']
        classifier = request.form['classifier']
        feature_selection = request.form['feature_selection']

        if classifier == 'Multinomial Naive Bayes':
            if feature_selection == 'PSO':
                result = pso_multinomial_naive_bayes(file_path)
            else:
                result = multinomial_naive_bayes(file_path)

        elif classifier == 'Bernoulli Naive Bayes':
            if feature_selection == 'PSO':
                result = pso_bernoulli_naive_bayes(file_path)
            else:
                result = bernoulli_naive_bayes(file_path)

        elif classifier == 'Gaussian Naive Bayes':
            if feature_selection == 'PSO':
                result = pso_gaussian_naive_bayes(file_path)
            else:
                result = gaussian_naive_bayes(file_path)

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

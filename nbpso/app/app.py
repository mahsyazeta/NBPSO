from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import preprocess_data
from models.naive_bayes import classify, predict_text
from models.pso import optimize_feature_selection

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

uploaded_file_path = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_path
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        flash('File successfully uploaded')
        return redirect(url_for('view_data'))

@app.route('/view_data')
def view_data():
    if uploaded_file_path:
        df = pd.read_csv(uploaded_file_path)
        tables = [df.to_html(classes='data')]
        return render_template('view_data.html', tables=tables)
    else:
        flash('No file uploaded yet')
        return redirect(url_for('index'))

@app.route('/preprocess')
def preprocess():
    if uploaded_file_path:
        df = pd.read_csv(uploaded_file_path)
        preprocessed_df = preprocess_data(df)
        tables = [preprocessed_df.to_html(classes='data')]
        return render_template('preprocessing.html', tables=tables)
    else:
        flash('No file uploaded yet')
        return redirect(url_for('index'))

@app.route('/classify', methods=['GET', 'POST'])
def classify_data():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        use_optimization = request.form.get('use_optimization') == 'on'
        if uploaded_file_path:
            df = pd.read_csv(uploaded_file_path)
            preprocessed_df = preprocess_data(df)
            if use_optimization:
                best_params = optimize_feature_selection(preprocessed_df, model_type)
            else:
                best_params = None
            results_df, avg_conf_matrix, avg_metrics, sentiment_dist_path = classify(preprocessed_df, model_type, best_params)
            
            # Combine classification results with original data
            combined_df = pd.DataFrame({
                'Text Tweet': preprocessed_df['Text Tweet'],  # Adjust this to your actual preprocessed text column name
                'Sentiment (Manual)': preprocessed_df['Sentiment'],
                'Sentiment (Model)': results_df['Predicted Sentiment'].apply(lambda x: 'positive' if x == 1 else 'negative')
            })
            return render_template('classify_form.html', classification_results=combined_df, metrics=avg_metrics, sentiment_dist_path=sentiment_dist_path, tables=[combined_df.to_html(classes='data')], titles=combined_df.columns.values)
        else:
            flash('No file uploaded yet')
            return redirect(url_for('index'))
    return render_template('classify_form.html', metrics=None)  # Default value for metrics when initially rendering the page



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        if uploaded_file_path:
            df = pd.read_csv(uploaded_file_path)
            preprocessed_df = preprocess_data(df)
            prediction = predict_text(text, preprocessed_df)
        else:
            flash('No file uploaded yet')
            return redirect(url_for('index'))
    return render_template('predict_form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

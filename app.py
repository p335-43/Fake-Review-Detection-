from flask import Flask, render_template, request
from joblib import load
import os

app = Flask(__name__)

# Define text_process before loading the model
def text_process(review):
    """
    Tokenizes text and removes stop words and punctuation.
    Used in the CountVectorizer.
    """
    import string
    from nltk.corpus import stopwords
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load the trained SVM model from a file
MODEL_PATH = 'svm_model.joblib'
if os.path.exists(MODEL_PATH):
    svm_pipeline = load(MODEL_PATH)
else:
    raise FileNotFoundError(f"The model {MODEL_PATH} does not exist.")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        new_review = request.form.get('review', '')
        if new_review:  # Ensure some text was actually entered
            prediction = svm_pipeline.predict([new_review])[0]
            prediction = 'Original' if prediction == 'OR' else 'Computer-generated'
    return render_template('home.html', prediction=prediction)

if __name__ == "__main__":
    app.run(port=5002)
from flask import Flask, render_template, request, jsonify
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    # review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    # for i in review_vect[0]:
    #     if i == 0:
    #         print('fake')
    print(review_vect[0].size)
    if model.predict(review_vect) == 0:
        prediction = 'FAKE' 
    elif model.predict(review_vect) ==1:
        prediction = 'REAL'
    else:
        prediction = 'INVALID'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()

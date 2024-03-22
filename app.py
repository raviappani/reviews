from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        model=joblib.load("logistic_regression.pkl")
        sentiment=model.predict([review])
        return render_template('result.html', review=review,sentiment=sentiment)
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)

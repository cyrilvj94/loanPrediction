import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from flask import redirect
import pickle


app = Flask(__name__)

with open('model.pkl','rb' ) as f:
    model = pickle.load(f)


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    #print(int_features)
    prediction = model.predict([int_features])[0]
    out = {0:"Customer will pay the loan in time", 1:'Default Customer'}
   # print(int_features)
    probability = int(model.predict_proba([int_features]).max()*100)
    out_text = f"Percentage Chance   : {probability}"
    return render_template('result.html', prediction_text=out[prediction], out=out_text)

@app.route('/home')
def return_home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    #app.run(host = 'localhost', port = 8000, debug = True)
    app.run(debug = True)
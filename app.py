import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
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
    print(int_features)
    prediction = model.predict([int_features])[0]
    out = {0:"Not a Defaulter Customer", 1:'Defaulter Customer'}
    return render_template('home.html', prediction_text=out[prediction])

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
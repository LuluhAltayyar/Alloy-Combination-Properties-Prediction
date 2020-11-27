import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
   return render_template('gp2.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':

      float_features = [float(x) for x in request.form.values()]
      final_features = [np.array(float_features)]
      result = model.predict(final_features)

      return render_template('gp2.html', result = result)

if __name__ == '__main__':
   app.run(debug = True)

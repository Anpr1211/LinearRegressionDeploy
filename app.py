import joblib 
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
model = joblib.load('finalized_model.pkl')

def predict(data):

    features = [float(x) for x in data.split()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    return prediction

@app.route('/', methods=['POST'])
def query():
   
    data = request.json['text']
    
    res = predict(data)

    return jsonify(res[0]) 


if __name__ == "__main__":
    app.run(debug=True)

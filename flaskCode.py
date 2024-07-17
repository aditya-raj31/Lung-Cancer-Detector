from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow import keras

app = Flask(_name_)
model = keras.models.load_model('lung_cancer_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
	if 'file' not in request.files:
		return 'No file uploaded'
	file = request.files['file']
	image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
	image = cv2.resize(image, (256, 256))
	image = np.expand_dims(image, axis=0)
	prediction = model.predict(image)
	prediction = np.argmax(prediction, axis=1)
	return jsonify({'prediction': prediction})

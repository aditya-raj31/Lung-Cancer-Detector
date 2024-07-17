import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

import cv2
import gc
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers


import warnings
warnings.filterwarnings('ignore')



from zipfile import ZipFile

data_path = 'archive.zip'

with ZipFile(data_path,'r') as zip:
  zip.extractall()
print('The data set has been extracted.')




path = 'lung_colon_image_set/lung_image_sets'
if os.path.exists(path):
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(classes)
else:
    print("Directory does not exist.")





IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 8

X = []
Y = []

for i, cat in enumerate(classes):
    images = glob(f'{path}/{cat}/*.jpeg')

    for image in images:
        img = cv2.imread(image)
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values

X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y,
                                                  test_size = SPLIT,
                                                  random_state = 2022)
print(X_train.shape, X_val.shape)



model = keras.models.Sequential([
	layers.Conv2D(filters=32,
				kernel_size=(5, 5),
				activation='relu',
				input_shape=(IMG_SIZE,
							IMG_SIZE,
							3),
				padding='same'),
	layers.MaxPooling2D(2, 2),

	layers.Conv2D(filters=64,
				kernel_size=(3, 3),
				activation='relu',
				padding='same'),
	layers.MaxPooling2D(2, 2),

	layers.Conv2D(filters=128,
				kernel_size=(3, 3),
				activation='relu',
				padding='same'),
	layers.MaxPooling2D(2, 2),

	layers.Flatten(),
	layers.Dense(256, activation='relu'),
	layers.BatchNormalization(),
	layers.Dense(128, activation='relu'),
	layers.Dropout(0.3),
	layers.BatchNormalization(),
	layers.Dense(3, activation='softmax')
])



model.summary()


model.compile(
	optimizer = 'adam',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
)


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if logs.get('val_accuracy') > 0.90:
			print('\n Validation accuracy has reached upto \
					90% so, stopping further training.')
			self.model.stop_training = True



es = EarlyStopping(patience=3,
				monitor='val_accuracy',
				restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_loss',
					patience=2,
					factor=0.5,
					verbose=1)



history = model.fit(X_train, Y_train,
					validation_data = (X_val, Y_val),
					batch_size = BATCH_SIZE,
					epochs = EPOCHS,
					verbose = 1,
					callbacks = [es, lr, myCallback()])


history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()
Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis=1)
Y_pred = np.argmax(Y_pred,axis=1)

metrics.confusion_matrix(Y_val, Y_pred)

print(metrics.classification_report(Y_val, Y_pred,
									target_names=classes))
									
							
model.save('lung_cancer_model.h5')
print("Model saved")
# Load the model
loaded_model = keras.models.load_model('lung_cancer_model.h5')
#Once you have loaded the saved model, you can use it to make predictions on new data without having to retrain the model.



#To test new data using the loaded model, you can use the following code:
# Make predictions on test data
test_images = []  # Replace with your test data
for image in test_images:
    img = cv2.imread(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_images.append(img)
test_images = np.asarray(test_images)



#the loaded model to make predictions on the new data.
# Make predictions
predictions = loaded_model.predict(test_images)
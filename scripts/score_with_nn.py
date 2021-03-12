import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.metrics import classification_report
import pickle
import time
import sys


model_path = sys.argv[1]
pickled_test_path = sys.argv[2]


print('== load in trained and saved model ===')

end_to_end_model =  tf.keras.models.load_model(model_path)





with open("/home/jupyter/sb-entity-classification/models/class_names.txt", "rb") as fp:  
    class_names = pickle.load(fp)

def get_prediction(string_input):
    probabilities = end_to_end_model.predict([[string_input]])
    return class_names[np.argmax(probabilities[0])]



print('== load in test set ===')

test = pd.read_pickle(pickled_test_path)
X_test = test['name'].tolist()


print('== start predicting on {} records ==='.format(len(X_test)))


start = time.time()
preds = list(map(get_prediction,X_test))
time_taken = time.time() - start
avg_time_p = time_taken / len(X_test)
print('Avg time per prediction {:.2f}s'.format(avg_time_p))
print('Total time for all {} forecasts: {:.2f}h'.format(len(X_test), time_taken / 3600))

print(classification_report(test['class_name'], preds, digits=3))

with open('/home/jupyter/sb-entity-classification/data/test_preds.pkl', "wb") as fp:   
    pickle.dump(preds, fp)








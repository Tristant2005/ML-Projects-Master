import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

colours = ['tab:blue','tab:orange','tab:green','tab:red']

#'https://www.kaggle.com/datasets/samybaladram/iris-dataset-extended'
'''
Species:
setosa = 0
versicolor = 1
virginica = 2

Soil:
Clay = 0
Loamy = 1
Sand = 2
'''

species_dict = {0: "Setosa", 1: "versicolor", 2: "virginica"}

elevation = float(input("Elevation(m)?: "))
soil = float(input("(0) Clay\n(1) Dirt\n(2) Sand\nSoil Type?: "))
sep_L = float(input("Lenth(cm) of Sepal?: "))
sep_W = float(input("Width(cm) of Sepal?: "))
ped_L = float(input("Length(cm) of Pedal?: "))
sep_W = float(input("Width(cm) of Pedal?: "))

#load the dataset
dataset = pd.read_csv('iris_extended.csv') 
# dataset.Legendary = dataset.Legendary.replace({True: 1, False: 0})
# dataset.Type1 = dataset.Type1.replace({"Water": 1, False: 0})

# print(dataset.Legendary)
def design_model(features):
    model = Sequential(name="Model")
    num_features = features.shape[1]

    #inputs (9 features)
    my_input = InputLayer(input_shape=(num_features,))
    model.add(my_input)
    #Hidden layer
    model.add(Dense(128, activation="relu"))
    #outputs 1 number
    # model.add(Dense(1))
    model.add(Dense(3, activation='softmax'))


    opt = Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy',  metrics=['accuracy'], optimizer=opt) #loss='mse',  metrics=['mae'], optimizer=opt) #

    return model

#choose first 7 columns as features
features = dataset.iloc[:,1:7] 
# #choose the final column for prediction
labels = dataset.iloc[:,0] 

#one-hot encoding for categorical variables
features = pd.get_dummies(features) 

#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.50, random_state=42) 

#print the number of features in the dataset
print("Number of features: ", features.shape[1]) 
#print the number of samples in the dataset
print("Number of samples: ", features.shape[0]) 
#see useful summary statistics for numeric features
# print(features.describe()) 

# #your code here below
# print(labels.shape)
# print(labels.describe()) 

# print(features_train.describe())
# print(features_test.describe())

model = design_model(features_train)
print(model.layers)

print(model.summary())

#my_model.fit(my_data, my_labels, epochs=50, batch_size=3, verbose=1)
hist = model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)

# plt.plot(hist.history['loss'],label='loss',marker='.')
# plt.plot(hist.history['val_loss'],label='val_loss',marker='.')
# plt.legend()
# plt.grid()
# plt.ylabel('mae')
# plt.xlabel('epoch')

val_loss, val_acc = model.evaluate(features_test, labels_test, verbose = 0)

print("Val loss: ", val_loss)
print("Val acc: ", val_acc)

# Make predictions
# input_data = np.array([[161.8, 2, 5.16, 3.41, 1.64, 0.26]])
input_data = np.array([[elevation, soil, sep_L, sep_W, ped_L, sep_W]])
predictions = model.predict(input_data)

print(predictions)
# Obtain predicted class labels
predicted_labels = np.argmax(predictions, axis=1)
# print(predicted_labels)

print(species_dict[predicted_labels[0]])




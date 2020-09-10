import numpy as np
import pandas as pd

HR_dataset = pd.read_csv('C:/Users/Prajit Vaghmaria/PycharmProjects/Employee_Retention_model/hr-comma-sepcsv/HR_comma_sep.csv')

print(HR_dataset.head())

categorical_features = ['department','salary']
HR_final = pd.get_dummies(HR_dataset,columns = categorical_features,drop_first=True)

from sklearn.model_selection import train_test_split

#drop left column
X = HR_final.drop(['left'],axis=1).values
y = HR_final['left'].values

X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#####Step 3  - Transform Data
##Scale your dataset for efficient computations , use StandardScaler from sklearn package

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
print(X_test.shape)
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test,y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
print(X_train.shape)
print(X_test.shape)

#####Build the neural network

import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Flatten

classifier = Sequential()
classifier.add(LSTM(32,input_shape = (X_train.shape[1], 1),activation='relu',return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(LSTM(32,activation='relu'))
classifier.add(Dropout(0.2))
#classifier.add(Flatten())
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics = ["accuracy"])
classifier.fit(X_train,y_train,batch_size=10,epochs = 10)
classifier.summary()

#### Making Predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
print(y_pred)

####Checking the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

### Single prediction
#single_new_feature = np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])
#transform = sc.transform(single_new_feature)
#pred_new_value = classifier.predict(single_new_feature)
#print(pred_new_value)


#new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))
new_pred = classifier.predict(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]]).reshape(1,18,1))
#new_pred2 = new_pred.reshape(1,18,1)
#new_pred3 = classifier.predict(new_pred2)
print(new_pred)
#print(new_pred.shape)
#print(new_pred3)
#new_pred = np.reshape(new_pred, (new_pred.shape[0],new_pred.shape[1],1)) # np.reshape(samples,timesteps,features)
#print(new_pred)
#print(new_pred.shape[0])
#print(new_pred.shape[1])

##add threshold
#new_pred = (new_pred > 0.5)
#print(new_pred)
### this threshold indicates that where the probability is above 50% an employee will leave the company


### improve model accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def make_classifier():
    classifier = Sequential()
    classifier.add(LSTM(32, input_shape=(X_train.shape[1],1), activation='relu', return_sequences=True))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(32, activation='relu'))
    classifier.add(Dropout(0.2))
   # classifier.add(Flatten())
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.summary()
    return classifier



classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=50)

accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs = 1)

#compute mean of the accuracies
mean = accuracies.mean()
print(mean)

#compute the variance of the accuracies
variance = accuracies.var()
print(variance)

from sklearn.model_selection import GridSearchCV

def make_classifier():

    classifier = Sequential()
    classifier.add(LSTM(32, input_shape=(X_train.shape[1],1), activation='relu', return_sequences=True))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(32, activation='relu'))
    classifier.add(Dropout(0.2))
    #classifier.add(Flatten())
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.summary()
    return classifier

classifier = KerasClassifier(build_fn = make_classifier)

params = {
    'batch_size':[20,35],
    'epochs':[20,30],
    'optimizer':['adam','rmsprop']
}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=2)

grid_search = grid_search.fit(X_train,y_train)

best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_


print(best_param)

print(best_accuracy)



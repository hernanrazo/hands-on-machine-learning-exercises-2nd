import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score


#get data
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

#train the 5-classifier
y_train5 = (y_train ==5)
y_test5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train5)

#use 5-classifier to predict. should return true
print(sgd_clf.predict([x[0]]))

#cross validate
cross_val = cross_val_score(sgd_clf, x_train, y_train5, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train5, cv=3)

#get confusion matrix
con_matrix = confusion_matrix(y_train5, y_train_pred)

#get precision and recall and f1 score
precision = precision_score(y_train5, y_train_pred)
recall = recall_score(y_train5, y_train_pred)
f1 = f1_score(y_train5, y_train_pred)
print('Confusion Matrix:')
print(con_matrix)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)


#multiclass classification using SVM
svm_clf = SVC()
svm_clf.fit(x_train, y_train)
print(svm_clf.predict([x[0]]))

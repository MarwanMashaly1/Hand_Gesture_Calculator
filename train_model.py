import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

X = np.load('X.npy')
y = np.load('y.npy')

n_per_class = 150
n_train = 100
n_test = 50

X_train, X_test, y_train, y_test = [], [], [], []
for label in np.unique(y):
    idx = np.where(y == label)[0]
    X_train.append(X[idx[:n_train]])
    X_test.append(X[idx[n_train:n_train+n_test]])
    y_train.append(y[idx[:n_train]])
    y_test.append(y[idx[n_train:n_train+n_test]])

X_train = np.concatenate(X_train)
X_test = np.concatenate(X_test)
y_train = np.concatenate(y_train)
y_test = np.concatenate(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
joblib.dump(clf, 'gesture_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
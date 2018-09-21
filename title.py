import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def read_data(path):
    db = pd.read_csv(path, sep=";")

    # delete time
    db = db.drop(['PurchaseDate'], axis=1)

    # oneHot cities and group name
    # 1260 городов - delete:)
    db = db.drop(['City'], axis=1)

    one_hot_2 = pd.get_dummies(db['GroupTNName'])
    db = db.join(one_hot_2)
    db = db.drop(['GroupTNName'], axis=1)

    X = db.values
    # from ['12640,00' '5023,00' '7815,00' ... '655,00' '10466,00' '341,10'] to [12640.0 5023.0 7815.0 ... 655.0 10466.0 341.1]
    X[:, 1] = [float(x[:x.find(',')] + '.' + x[x.find(',') + 1:]) for x in X[:, 1]]

    y = [random.choice([1, 0]) for _ in range(X.shape[0])]
    return X, np.asarray(y)


def run_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Classifier
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print("Accuracy of Logistic Ragression is", accuracy_score(y_test, y_pred) * 100)

    # Random Forest
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Accuracy of Random Forest is", accuracy_score(y_test, y_pred) * 100)

    # Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy of Decision Tree is", accuracy_score(y_test, y_pred) * 100)


def main():
    X, y = read_data("purchase_test.csv")
    run_models(X, y)

start = time.time()
main()
print("Time to do it all =", (time.time() - start) / 60, "minutes")
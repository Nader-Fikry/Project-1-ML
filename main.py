from sklearn.naive_bayes import GaussianNB  # import Gaussian Naive Bayes model
from sklearn import datasets  # import datasets from sklearn package
import matplotlib.pyplot as plt  # import pyplot from matplotlib


def train_validate_test_split(data, labels, test_ratio = 0.3, val_ratio = 0.3):  # split function to split the data
    train = data[90:]  # 40% for train set
    train_l = labels[90:]  # 40% for train label

    val = data[45:90]  # 30% for validation set
    val_l = labels[45:90]  # 30% for validation label

    test = data[0:45]  # 30% for test set
    test_l = labels[0:45]  # 30% for test label

    return train, train_l, val, val_l, test, test_l  # return the data after splitting it

def accuracy(test, predict):  # function to calculate accuracy
    right = 0  # record to the right prediction
    for i in range(len(test)):  # loop throw the test label
        if test[i] == predict[i]:  # if it is the right answer
            right += 1  # increment by one
    print("Accuracy of the model is: ", right / float(len(test)) * 100.0)


iris = datasets.load_iris()  # load iris dataset in iris object
data = iris.data  # load iris features in x_data
target = iris.target  # load iris targets in x_targets

#  return the values of split function
tr, tr_l, va, va_l, te, te_l = train_validate_test_split(data, target, test_ratio = 0.3, val_ratio = 0.3)

Bayes = GaussianNB()  # create an object of GaussianNB model
Bayes.fit(tr, tr_l)  # train the model with training set
Bayes.fit(va, va_l)  # optimize the algorithm by another training throw validation set
y = Bayes.predict(te)  # test the model throw test set


accuracy(te_l, y)  # print the accuracy of the model

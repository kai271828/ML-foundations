import csv
import random
import numpy as np

with open('data2_train.csv', newline='', encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile)
    data = [row for row in rows]
    X_train = np.array([[1.0] + list(map(float, row[0:-1])) for row in data])
    Y_train = np.array([int(row[-1]) for row in data])

with open('data2_test.csv', newline='', encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile)
    data = [row for row in rows]
    X_test = np.array([[1.0] + list(map(float, row[0:-1])) for row in data])
    Y_test = np.array([int(row[-1]) for row in data])


def error(W, X, Y):
    temp = np.dot(X, W) * Y
    e = 0
    for ele in temp:
        if ele <= 0:
            e += 1

    return e


W = np.zeros(X_train.shape[1])
W_hat = np.zeros(X_train.shape[1])
min_error = error(W_hat, X_train, Y_train)
total_error = 0

for exp in range(2000):
    random.seed(random.randint(0, 20000))
    for t in range(100):
        n = random.randint(0, 399)
        W = W + np.dot(X_train[n].T, Y_train[n])
        cur_error = error(W, X_train, Y_train)
        if cur_error < min_error:
            W_hat = W.copy()
            min_error = cur_error

    total_error += error(W_hat, X_test, Y_test) / 500.0

print(total_error / 2000.0)

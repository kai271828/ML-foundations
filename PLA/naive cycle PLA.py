import csv
import numpy as np

with open('data1.csv', newline='', encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile)
    data = [row for row in rows]
    X = np.array([[1.0] + list(map(float, row[0:-1])) for row in data])
    Y = np.array([int(row[-1]) for row in data])

W = np.zeros(X.shape[1])
t = 0

for i in range(X.shape[0]):
    product = np.dot(X[i], W)
    if product * Y[i] <= 0:
        W = W + np.dot(X[i].T, Y[i])
        t += 1

print(t)

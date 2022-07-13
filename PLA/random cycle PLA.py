import csv
import random
import numpy as np


def determine_order(o):
    for j in range(400):
        random_index = random.randint(0, 399)
        temp = o[j]
        o[j] = o[random_index]
        o[random_index] = temp


with open('data1.csv', newline='', encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile)
    data = [row for row in rows]
    X = np.array([[1.0] + list(map(float, row[0:-1])) for row in data])
    Y = np.array([int(row[-1]) for row in data])


total = 0
eta = 0.5
order = [num for num in range(X.shape[0])]
for exp in range(2000):
    determine_order(order)
    W = np.zeros(X.shape[1])
    t = 0

    for i in order:
        product = np.dot(X[i], W)
        if product * Y[i] <= 0:
            W = W + np.dot(X[i].T, Y[i]) * eta
            t += 1
    total += t

print(total / 2000.0)

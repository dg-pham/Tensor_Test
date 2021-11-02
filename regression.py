import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv('Data/student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# print(data.head())

predict = 'G3'

X = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# best = 0
# for _ in range(30):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#
#     reg = LinearRegression()
#     reg.fit(X_train, y_train)
#
#     acc = reg.score(X_test, y_test)
#     if acc > best:
#         best = acc
#         with open('Models/student_model.pickle', 'wb') as f:
#             pickle.dump(reg, f)

# with open('Models/student_model.pickle', 'rb') as read_pickle:
#     reg = pickle.load(read_pickle)

# print(reg.coef_)
# print(reg.intercept_)

# print(best)

# predictions = reg.predict(X_test)
# for i in range(len(predictions)):
#     print(predictions[i], X_test[i], y_test[i])

p = 'G2'
style.use('ggplot')
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()

# -----------------------------------------------------------------------------------

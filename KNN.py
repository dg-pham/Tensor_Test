import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('data/car.data')
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
doors = le.fit_transform(list(data['doors']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# acc = knn.score(X_test, y_test)
# print(acc)

classes = ['acc', 'good', 'unacc', 'vgood']
predicted = knn.predict(X_test)

for i in range(len(predicted)):
    # print(classes[predicted[i]], X_test[i], classes[y_test[i]])
    n = knn.kneighbors([X_test[i]], n_neighbors=20)
    print(n)
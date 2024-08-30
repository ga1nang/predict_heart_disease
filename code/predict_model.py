import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

#compare different model
models = []
name_model = ["KNN", "SVC", "Naive Bayes", "Decision Tree", 'Random Forest', 'Adaboost', 'Gradient Boost', 'XGboost', 'Stacking']
acc_model = []
time_model = []

#prepare data
df = pd.read_csv('dataset\\cleaned_cleveland.csv', header=None)
X = df.iloc[1:, :-1].values
y = df.iloc[1:, -1].values.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)


#calculate accuract
def acc(y_true, y_pred):
    return np.round(accuracy_score(y_true, y_pred), 2)


#A form for the different models
def form_model(model, model_name):
    start = time()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    accuracy = acc(y_train, pred_train)
    process_time = str(round(time() - start,5))
    print(model_name, "accuracy for train =", accuracy, "and accuracy for test =", acc(y_test, pred_test))
    print('Time to process', model_name, 'is', process_time)
    acc_model.append(accuracy)
    time_model.append(process_time)




#KNN model
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
form_model(knn, "KNN")
models.append(knn)


#SVM model
svc = SVC(kernel='rbf', random_state=29)
form_model(svc, "SVC")
models.append(svc)


#Naive Bayes model
nb = GaussianNB()
form_model(nb, "Naive Bayes")
models.append(nb)


#Decisiton Tree model
dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2)
form_model(dt, "Decision Tree")
models.append(dt)


#Random Forest
rfc = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2, n_estimators=10, random_state=29)
form_model(rfc, 'Random Forest')
models.append(rfc)


#Adaboost
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME')
form_model(abc, 'Adaboost')
models.append(abc)


#GradientBoost
gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth=3, random_state=29)
form_model(gbc, 'Gradient Boost')
models.append(gbc)


#XGboost
xgb = XGBClassifier(objective='binary:logistic', random_state=29, n_estimators=100) 
form_model(xgb, 'XGboost')
models.append(xgb)


#Stacking with XGboost is final estimator
clf =[]
for i in range(0, len(models)-1):
    clf.append((name_model[i], models[i]))

stacking = StackingClassifier(estimators=clf, final_estimator=XGBClassifier())
form_model(stacking, 'Stacking')


#plot compare graph
# Plotting the accuracy
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.bar(name_model, acc_model, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')

# Plotting the processing time
plt.subplot(1, 2, 2)
plt.bar(name_model, time_model, color='salmon')
plt.title('Model Processing Time Comparison')
plt.ylabel('Processing Time (s)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('graph\\compare_models.png')

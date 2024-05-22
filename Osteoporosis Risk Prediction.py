import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("M:/tutorials/DATA SCIENCE/Osteoporosis Risk Prediction/osteoporosis.csv")
df.head()

df.shape
df.info()
<<<<<<< HEAD
=======

df.isna().sum()

columns_with_missing_values = df.columns[df.isnull().any()]

#missing value percentage
print("Missing value percentage")
for column in columns_with_missing_values:
    print(column,":",df[column].isnull().sum()/df.shape[0]*100)

#replace missing values with "None"
df.fillna("None",inplace=True)

df = df.drop(['Id'], axis=1)

#value counts of categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(df[column].value_counts())


df.describe()

#pie chart for the target variable (Osteoporosis)
plt.figure(figsize = (5,5))
df['Osteoporosis'].value_counts().plot.pie(autopct = '%1.1f%%').set_title('Osteoporosis')
# perfectly balanced with 50% of the patients having osteoporosis and 50% not having osteoporosis, which means that the dataset is
# not biased towards any class

plt.figure(figsize = (5,5))
df[df['Osteoporosis'] == 1]['Age'].plot.hist(bins = 30, alpha=0.5, color='blue', label = 'Osteoporosis = 1')
df[df['Osteoporosis'] == 0]['Age'].plot.hist(bins = 30, alpha=0.5, color='red', label = 'Osteoporosis = 0')
#legends and title
plt.legend()
plt.xlabel('Age')
plt.title('Osteoporosis by Age')
# there is significant risk of osteoporosis in patients of all ages but patients between the ages 20 to 40 have significantly much lower risk of
# osteoporosis. This highlights that fact that younger patients are less likely to have osteoporosis

sns.countplot(x = 'Gender', data = df, hue = 'Osteoporosis').set_title('gender vs osteoporosis')
# there is no concrete relationship between gender and the risk of osteoporosis

sns.countplot(x = 'Hormonal Changes', data = df, hue = 'Osteoporosis').set_title('Hormonal Changes vs osteoporosis')
# The graph shows that patients who have undergone hormonal changes have a higher
# risk of osteoporosis than those who have not undergone hormonal changes. This
# indicates that hormonal changes can be a significant risk factor for osteoporosis. This
# highlights that our hormones contribute in making our bones strong

sns.countplot(x = 'Family History', data = df, hue = 'Osteoporosis').set_title('Family History vs osteoporosis')
#there is not much differnece in both cases regarding the risk of osteoporosis. Therefore, family history couldn;t be considered a predictor for osteoporosis.

sns.countplot(x = 'Race/Ethnicity', data = df, hue = 'Osteoporosis').set_title('Race/Ethicity vs osteoporosis')
#  the risk of osteoporosis is almost similar with no concrete relationship between the race and risk of osteoporosis.

sns.countplot(x = 'Body Weight', data = df, hue = 'Osteoporosis').set_title('Body weight vs osteoporosis')
#  lower body weight have a higher risk of osteoporosis than those with higher body weight. 

fig, ax = plt.subplots(1,2,figsize = (20,6))
sns.countplot(x = 'Calcium Intake',data = df, ax = ax[0], hue= 'Osteoporosis')#.set_titile('calcium vs osteoporosis')
sns.countplot(x = 'Vitamin D Intake', data = df, ax = ax[1], hue= 'Osteoporosis')#.set_titile('Vitamin D vs osteoporosis')

# patients with lower calcium and vitamin D levels have a higher risk of osteoporosis than those with higher
# calcium and vitamin D levels. This indicates that nutrition can be a significant risk factor for osteoporosis. This highlights that our nutrition contributes in making our bones strong.

sns.countplot(data = df, x = 'Physical Activity', hue = 'Osteoporosis').set_title('Physical Activity vs osteoporosis')

fig, ax = plt.subplots(1,2,figsize = (20,6))
sns.countplot(x = 'Smoking',data = df, ax = ax[0], hue= 'Osteoporosis')#.set_titile('calcium vs osteoporosis')
sns.countplot(x = 'Alcohol Consumption', data = df, ax = ax[1], hue= 'Osteoporosis')#.set_titile('Vitamin D vs osteoporosis')
# This indicates that smoking and alcohol consumption are not significant risk factors for osteoporosis.

fig, ax = plt.subplots(1,2,figsize = (20,6))
sns.countplot(x = 'Medical Conditions',data = df, ax = ax[0], hue= 'Osteoporosis')#.set_titile('calcium vs osteoporosis')
sns.countplot(x = 'Medications', data = df, ax = ax[1], hue= 'Osteoporosis')#.set_titile('Vitamin D vs osteoporosis')

sns.countplot(x = 'Prior Fractures', data = df, hue = 'Osteoporosis').set_title('Body weight vs osteoporosis')
# no concrete relationship between the prior incident of fractures and risk of osteoporosis

# In[]
#columns for label encoding
cols = df.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
    print(col, ":",df[col].unique())

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Osteoporosis',axis=1), df['Osteoporosis'], test_size=0.30, random_state=101)

# In[] logistic regresstion

from sklearn.linear_model import LogisticRegression

# Hyperparameter Tuning using GridSearchCV
logmodel = LogisticRegression()

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,10, 10, 100, 1000],
              'penalty': ['l1','l2'],
              'solver': ['liblinear'],
                'max_iter':[100, 1000, 2500, 5000],
                'multi_class': ['auto', 'ovr'],
                'random_state': [0, 42, 101]}

# grid search object
grid = GridSearchCV(logmodel, param_grid, refit = True, verbose=3,cv =5,n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)

logmodel = LogisticRegression(C = 0.1, max_iter=100, penalty = 'l2', random_state=0, solver='liblinear', multi_class='auto')
logmodel.fit(X_train, y_train)
print('train accuracy: ', logmodel.score(X_train, y_train))

lr_pred = logmodel.predict(X_test)
# In[] Random Farest

from sklearn.ensemble import RandomForestClassifier

# creating randomfarest object
rfc = RandomForestClassifier()

param_grid = { 'criterion':['gini', 'entropy'],
              'max_depth': [10, 20, 30],
              'min_samples_split': [2,5,10],
              'min_samples_leaf':[2,5,10],
              'random_state': [0,42,101]}
# gtid searh object
grid = GridSearchCV(rfc, param_grid, refit = True, verbose = 3,  cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)
print(grid.best_params_)

rfc = RandomForestClassifier(criterion='gini', max_depth= 20, min_samples_leaf=2, min_samples_split=2, random_state=42)
rfc.fit(X_train, y_train)
print('train accuracy: ', rfc.score(X_train, y_train))
rfc_pred = rfc.predict(X_test)
# In[] decision tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

param_grid = {'criterion':['gini', 'entropy'],
              'max_depth':[10,20,30],
              'min_samples_split':[2,5,10],
              'min_samples_leaf': [2,5,10],
              'random_state': [0,42,101]}
grid = GridSearchCV(dtree, param_grid, cv = 5, n_jobs = -1)
grid.fit(X_train,y_train)
print(grid.best_params_)

dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, min_samples_leaf = 10, min_samples_split = 2, random_state =  0)
dtree.fit(X_train,y_train)
print('train accuracy: ', dtree.score(X_train,y_train))
dtree_pred = dtree.predict(X_test)
# In[] SVM
from sklearn.svm import SVC

svc = SVC()

param_grid = {'C': [0.1, 1, 10, 100],
              'degree': [2, 3, 4, 5],
              'gamma': ['scale', 'auto'],
              'random_state': [0,42,101]}

grid = GridSearchCV(svc, param_grid, verbose=3, cv = 5, n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)

svc = SVC(C = 1, degree = 2, gamma = 'auto', random_state = 0)
svc.fit(X_train, y_train)
print('train accuracy: ', svc.score(X_train, y_train))
svc_pred = svc.predict(X_test)
# In[] model evaluation
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(2,2,figsize = (15,15))
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot = True, ax = ax[0,0],fmt = 'g').set_title('logistic regression')

cm = confusion_matrix(y_test, rfc_pred)
sns.heatmap(cm, annot = True, ax = ax[0,1], fmt = 'g').set_title('random farest regression')

cm = confusion_matrix(y_test, dtree_pred)
sns.heatmap(cm, annot = True, ax = ax[1,0], fmt = 'g').set_title('decicion tree regression')

cm = confusion_matrix(y_test, svc_pred)
sns.heatmap(cm, annot = True, ax = ax[1,1], fmt = 'g').set_title('svm regresstio')

# accuracy bar chart
from sklearn.metrics import accuracy_score
models = ['Logistic regression', 'random farest', 'decision tree', 'svm']
model_accuracy = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rfc_pred), accuracy_score(y_test, dtree_pred), accuracy_score(y_test, svc_pred)]
plt.figure(figsize=(10,5))
sns.barplot(x = models, y = model_accuracy).set_title('model accuracy')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
models = ['Logistic regression', 'random farest', 'decision tree', 'svm']
fig, ax = plt.subplots(2,2,figsize = (15,15))
mae = [mean_absolute_error(y_test, lr_pred), mean_absolute_error(y_test, rfc_pred), mean_absolute_error(y_test, dtree_pred), mean_absolute_error(y_test, svc_pred)]
mse = [mean_squared_error(y_test, lr_pred), mean_squared_error(y_test, rfc_pred), mean_squared_error(y_test, dtree_pred), mean_squared_error(y_test, svc_pred)]
r2 = [r2_score(y_test, lr_pred), r2_score(y_test, rfc_pred), r2_score(y_test, dtree_pred), r2_score(y_test, svc_pred)]

sns.barplot(x = models, y = mae, ax = ax[0,0]).set_title('MAE')

sns.barplot(x = models, y = mse, ax = ax[0,1]).set_title('MSE')

sns.barplot(x = models, y = r2, ax = ax[1,0]).set_title('R2')




>>>>>>> 1c19853 (delete the file)

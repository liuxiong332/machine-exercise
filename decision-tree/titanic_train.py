import numpy as np
import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def get_title(name):
    match = re.search(r' (\w+)\.', name)
    return match.group(1) if match else ''

def load_dataset():
    train_data = pd.read_csv('titanic_train.csv')
    test_data = pd.read_csv('titanic_test.csv')

    sex_map = {'male': 0, 'female': 1}
    train_data['Sex'] = train_data['Sex'].map(sex_map)
    test_data['Sex'] = test_data['Sex'].map(sex_map)

    # 填充Embarked空白，并把字符串值转化成 数值 
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    train_data['Embarked'] = train_data['Embarked'].fillna('S').map(embarked_map)
    test_data['Embarked'] = test_data['Embarked'].fillna('S').map(embarked_map)

    # 使用年龄的平均值 填充 Age
    mean_age = train_data['Age'].mean()
    train_data['Age'] = train_data['Age'].fillna(mean_age)
    test_data['Age'] = test_data['Age'].fillna(mean_age)

    test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].mean())
    features1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    print('apply featuers1:')
    # logistic_train(train_data, features1)
    # decision_tree_train(train_data, features1)
    rfc_train(train_data, features1)
    rfc_test(train_data, features1, test_data)
    # gbc_train(train_data, features1)
    # ada_boost_train(train_data, features1)

    return
    # Generating a familysize column
    train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]

    # The .apply method generates a new series
    train_data["NameLength"] = train_data["Name"].apply(lambda x: len(x))
    train_data['Title'] = train_data['Name'].apply(get_title)
    title_map = {
        'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Don': 5, 'Rev': 6, 'Dr': 7, 'Mme': 8, 'Ms': 2,
        'Major': 9, 'Lady': 10, 'Sir': 11, 'Mlle': 12, 'Col': 13, 'Capt': 14, 'Countess': 15, 'Jonkheer': 16
    }
    train_data['Title'] = train_data['Title'].map(title_map)
    features2 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']        
    print('apply features2:')
    logistic_train(train_data, features2)
    decision_tree_train(train_data, features2)
    rfc_train(train_data, features2)
    gbc_train(train_data, features2)
    ada_boost_train(train_data, features2)
    
    stack_train(train_data, ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"], ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",])

def logistic_train(train_data, train_features):
    sm = SMOTE()
    x_train = train_data[train_features]
    y_train = train_data['Survived']
    x_train, y_train = sm.fit_sample(x_train, y_train)

    gcv = GridSearchCV(estimator=LogisticRegression(penalty='l2'), param_grid={'C': [0.01, 0.1, 1, 10, 100]}) 
    gcv.fit(x_train, y_train)
    print('Logistic Regression accuracy %f' % gcv.best_score_) 

def decision_tree_train(train_data, train_features):
    dt = DecisionTreeClassifier(max_depth=10)    
    scores = cross_val_score(dt, train_data[train_features], train_data['Survived'], cv=5)
    print('DecisionTree Classfier accuracy %f' % scores.mean()) 

def rfc_train(train_data, train_features):
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    scores = cross_val_score(rfc, train_data[train_features], train_data['Survived'], cv=5)
    print('DecisionTree Classfier accuracy %f' % scores.mean()) 
    
def rfc_test(train_data, train_features, test_data):
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc.fit(train_data[train_features], train_data['Survived'])
    predict_vals = rfc.predict(test_data[train_features])
    pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predict_vals}).to_csv('result.csv', index=False)

def gbc_train(train_data, train_features):
    gbc = GradientBoostingClassifier()
    scores = cross_val_score(gbc, train_data[train_features], train_data['Survived'], cv=5)    
    print('GradientBoosting Classfier accuracy %f' % scores.mean())

def ada_boost_train(train_data, train_features):
    train_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']    
    abc = AdaBoostClassifier(n_estimators=100)    
    scores = cross_val_score(abc, train_data[train_features], train_data['Survived'], cv=5)
    print('AdaBoost Classifier accuracy %f' % scores.mean())

def stack_train(train_data, features1, features2):
    kf = StratifiedKFold(5)
    algs = [
        (LogisticRegression(), features1),
        (RandomForestClassifier(), features2) 
    ]

    scores = []
    for train_index, test_index in kf.split(train_data[features1], train_data['Survived']):
        trains = train_data.iloc[train_index]
        tests = train_data.iloc[test_index]
        probas = []
        for alg, features in algs:
            alg.fit(trains[features], trains['Survived'])
            probas.append(alg.predict_proba(tests[features])[:, 1])
        score = accuracy_score(tests['Survived'], (probas[0] + probas[1] * 3) / 4 > 0.5)
        scores.append(score)
    
    print('Stack Classifier accuracy %f' % np.mean(scores))
    
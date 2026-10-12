# Archive: Homework3.py
# Author: @mzc07
# Description: Classification practice using a course_lead_scoring dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score

# 0. Load dataset
df_lead = pd.read_csv('data/course_lead_scoring_2026.csv')
target = 'converted'

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']


# Data preparation
print('Missing Values')
print(df_lead.isnull().sum())

df_lead[categorical] = df_lead[categorical].fillna('NA')
df_lead[numerical] = df_lead[numerical].fillna(0.0)


def train_model(train, val, y_train, y_val, cat, num, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train[cat + num].to_dict(orient='records'))
    X_val = dv.transform(val[cat + num].to_dict(orient='records'))

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    return accuracy_score(y_val, pred)


# 1. Mode of industry (1 point)
print('\nQ1. Mode of industry')
print(df_lead['industry'].mode()[0])

# 2. Correlation matrix (1 point)
print('\nQ2. Correlation matrix')
corr = df_lead[numerical].corr()
print(corr)
pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count'),
]
for a, b in pairs:
    print(f'{a} - {b}: {corr.loc[a, b]:.4f}')

# Split data
df_full_train, df_test = train_test_split(df_lead, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

y_train = df_train[target].values
y_val = df_val[target].values

df_train = df_train.drop(columns=target)
df_val = df_val.drop(columns=target)

# 3. Mutual information (1 point)
print('\nQ3. Mutual information')
for col in categorical:
    score = mutual_info_score(y_train, df_train[col])
    print(f'{col}: {round(score, 2)}')

# 4. Logistic regression (1 point)
print('\nQ4. Accuracy')
base_acc = train_model(df_train, df_val, y_train, y_val, categorical, numerical)
print(round(base_acc, 2))

# 5. Feature elimination (1 point)
print('\nQ5. Feature elimination')
features = categorical + numerical
for f in features:
    cat = [c for c in categorical if c != f]
    num = [n for n in numerical if n != f]
    acc = train_model(df_train, df_val, y_train, y_val, cat, num)
    print(f'without {f}: acc={acc:.4f}, diff={base_acc - acc:.4f}')

# 6. Regularization (1 point)
print('\nQ6. Regularization')
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    acc = train_model(df_train, df_val, y_train, y_val, categorical, numerical, C=C)
    print(f'C={C}: {round(acc, 3)}')
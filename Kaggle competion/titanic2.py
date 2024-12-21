import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
    
    
# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Impute missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

# Ensure consistent dummy variables
combined = pd.concat([train, test], keys=['train', 'test'])
combined = pd.get_dummies(combined, columns=['Sex', 'Embarked'], drop_first=True)

df_train = combined.loc['train'].drop(['Survived'], axis=1)
df_train_y = train['Survived']
df_test = combined.loc['test'].drop(['Survived'], axis=1)
df_train.drop('Name',axis=1,inplace=True)
df_test.drop('Name',axis=1,inplace=True)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_y, test_size=0.3, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Build and test the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train_balanced, y_train_balanced)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Test the model on unseen data
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
pred = model.predict(df_test)

output = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": pred
})

output.to_csv("Predictions.csv", index=False)
print("Predictions saved to predictions.csv")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('train.csv')

# 1. Encode 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 2. Encode 'Embarked'
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# If there are missing values in 'Embarked', fill with the most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 3. Encode 'Pclass' (Optional - treat as categorical if needed)
df['Pclass'] = df['Pclass'].astype(int)

# 4. Handle 'Cabin'
# Extract the deck from the first letter of the 'Cabin' column
df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
# One-hot encode the Deck feature
df = pd.get_dummies(df, columns=['Deck'], prefix='Deck')
# Drop the original 'Cabin' column
df.drop('Cabin', axis=1, inplace=True)

# 5. Process 'Name'
# Extract title from the name
df['Title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
# Map uncommon titles to broader categories
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Countess': 'Nobility',
    'Lady': 'Nobility', 'Sir': 'Nobility', 'Capt': 'Officer', 'Jonkheer': 'Nobility',
    'Don': 'Nobility'
}
df['Title'] = df['Title'].map(title_mapping)
# One-hot encode the Title feature
df = pd.get_dummies(df, columns=['Title'], prefix='Title')
# Drop the raw 'Name' field
df.drop('Name', axis=1, inplace=True)

# 6. Handle 'Ticket'
# Extract prefix from the Ticket column
df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if x.isalpha() else 'None')
# One-hot encode the Ticket prefix
df = pd.get_dummies(df, columns=['TicketPrefix'], prefix='Ticket')
# Drop the raw 'Ticket' field
df.drop('Ticket', axis=1, inplace=True)

# 7. Feature engineering
# Create 'FamilySize' feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Create 'IsAlone' feature
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 8. Handle numerical features ('Age' and 'Fare')
# Fill missing values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
# Scale the numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])



#separate the data
X= df.drop('Survived',axis=1)
y = df['Survived']

#split the testing and traing data
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.3,random_state=42)

#build and train the model
model = RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train,y_train)

#test the model
y_pred= model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f"Model accuracy:{accuracy}")


# Preprocess the test dataset
test_df = pd.read_csv('test.csv')

# 1. Encode 'Sex'
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# 2. Encode 'Embarked'
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 3. Encode 'Pclass' (Optional)
test_df['Pclass'] = test_df['Pclass'].astype(int)

# 4. Handle 'Cabin'
test_df['Deck'] = test_df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
test_df = pd.get_dummies(test_df, columns=['Deck'], prefix='Deck')
test_df.drop('Cabin', axis=1, inplace=True)

# 5. Process 'Name'
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
test_df['Title'] = test_df['Title'].map(title_mapping)
test_df = pd.get_dummies(test_df, columns=['Title'], prefix='Title')
test_df.drop('Name', axis=1, inplace=True)

# 6. Handle 'Ticket'
test_df['TicketPrefix'] = test_df['Ticket'].apply(lambda x: x.split()[0] if x.isalpha() else 'None')
test_df = pd.get_dummies(test_df, columns=['TicketPrefix'], prefix='Ticket')
test_df.drop('Ticket', axis=1, inplace=True)

# 7. Feature engineering
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

# 8. Handle numerical features ('Age' and 'Fare')
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])  # Use the same scaler

# Align features with the training set (ensure matching columns)
test_df = test_df.reindex(columns=X_train.columns, fill_value=0)

# Make predictions
test_predictions = model.predict(test_df)

# Output predictions
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],  # Assuming the test file has an index or restore PassengerId column
    'Survived': test_predictions
})

# Save predictions to a CSV file
output.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")



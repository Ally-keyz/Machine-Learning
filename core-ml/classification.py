from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#load the data set
data = load_iris()
X,y = data.data , data.target

#split the data 
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Build and train the model
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#make predictions
y_pred = model.predict(X_test)

#check the accuracy
accuracy = accuracy_score(y_test,y_pred)

print(f"Model accuracy:{accuracy}")


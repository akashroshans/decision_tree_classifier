import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('dataset.csv')

label_encoder = LabelEncoder()
data['Education'] = label_encoder.fit_transform(data['Education'])
data['MaritalStatus'] = label_encoder.fit_transform(data['MaritalStatus'])
data['Buy'] = data['Buy'].map({'Yes': 1, 'No': 0})

X = data[['Age', 'Income', 'Education', 'MaritalStatus', 'Children']]
y = data['Buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
education_mapping = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}
marital_status_mapping = {
    'Single': 0,
    'Married': 1,
    'Divorced': 2
}
def make_prediction():
    age = int(input("Enter age: "))
    income = float(input("Enter income: "))
    education_input = input("Enter education level (High School, Bachelor, Master, PhD): ")
    marital_status_input = input("Enter marital status (Single, Married, Divorced): ")
    children = int(input("Enter number of children: "))
    education = education_mapping.get(education_input, -1)  
    marital_status = marital_status_mapping.get(marital_status_input, -1)
    if education == -1 or marital_status == -1:
        print("Invalid input for education or marital status.")
    else:
        person = pd.DataFrame([[age, income, education, marital_status, children]], 
                              columns=['Age', 'Income', 'Education', 'MaritalStatus', 'Children'])
        prediction = best_model.predict(person)
        if prediction[0] == 1:
            print("Prediction: The person will buy the product.")
        else:
            print("Prediction: The person will not buy the product.")
make_prediction()

# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Bring in the necessary libraries.

2. **Load the Dataset**:  
   Load the dataset into your environment.

3. **Data Preprocessing**:  
   Handle any missing data and encode categorical variables as needed.

4. **Define Features and Target**:  
   Split the dataset into features (X) and the target variable (y).

5. **Split Data**:  
   Divide the dataset into training and testing sets.

6. **Build Multiple Linear Regression Model**:  
   Initialize and create a multiple linear regression model.

7. **Train the Model**:  
   Fit the model to the training data.

8. **Evaluate Performance**:  
   Assess the model's performance using cross-validation.

9. **Display Model Parameters**:  
   Output the model’s coefficients and intercept.

10. **Make Predictions & Compare**:  
    Predict outcomes and compare them to the actual values. 

## Program:
```
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: R.Mushafina
RegisterNumber: 212224220067

import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load and prepare data
data= pd.read_csv('C:/Users/admin/Downloads/CarPrice_Assignment (1).csv')

# Simple preprocessing
data = data.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data,drop_first=True)

# 2.Split data
x=data.drop('price',axis=1)
y=data['price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# 3. Create and train model
model=LinearRegression()
model.fit(x_train,y_train)

# 4. Evaluate with cross-validation (simple version)
print('Name: R.Mushafina')
print('Reg. No: 212224220067')
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model,x,y,cv=5)
print("Fold R^2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R^2: {cv_scores.mean():.4f}")

# 5. Test set evaluation
y_pred = model.predict(x_test)
print("\n=== Test set Perfomance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")

# 6. Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="978" height="701" alt="image" src="https://github.com/user-attachments/assets/5695c691-ac02-4ef4-8f6c-95998ba8f7d8" />
<img width="1021" height="745" alt="image" src="https://github.com/user-attachments/assets/e1acd89d-2715-4b32-ae4e-667163111948" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.

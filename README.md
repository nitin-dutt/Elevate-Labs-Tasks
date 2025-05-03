Hello! I am uploading all the tasks in a single repo and am giving the readme for all the tasks here as I thought it would be easier to navigate between the tasks this way. I provided the readme for each task clearly seperated here. While writing the code, I divided the code into sectitons and I gave a brief description of what each section of the code is doing.

=========================================================================== Task 1: Data Cleaning & Preprocessing – Titanic Dataset =========================================================================

This repository contains the preprocessing pipeline for the Titanic dataset, focused on cleaning, handling missing values, encoding categorical variables, scaling features, and removing outliers.

## 📌 Objective
The goal of this task is to prepare raw Titanic data for machine learning by performing a full preprocessing workflow.

## 🧰 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Seaborn, Matplotlib
- scikit-learn

## 📊 Dataset Used
Source: [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

## ⚙️ Workflow & Steps
SECTION 1. Import Libraries
Loaded all necessary libraries for data processing and visualization.

SECTION 2. Load Dataset
Imported the Titanic dataset directly from a URL using `pandas.read_csv()`.
We can also import from our storage using 'pd.read_csv(#Path)' 

SECTION 3. Explore Dataset
Viewed the first few rows and checked for:
- Data types
- Missing values
- Basic structure using `.head()`, `.info()`, and `.isnull().sum()`

SECTION 4. Handle Missing Values
- Filled missing 'Age' values using 'median'
- Filled missing 'Embarked' values using 'mode'
- Dropped the 'Cabin' column due to excessive missing data

SECTION 5. Encode Categorical Columns
- Encoded 'Sex' and 'Embarked' using **Label Encoding**

SECTION 6. Feature Scaling
- Standardized `Age` and `Fare` using **StandardScaler**

SECTION 7. Detect Outliers
- Visualized outliers in `Age` and `Fare` using **boxplots**
- 
SECTION 8. Remove Outliers
- Used **IQR method** to filter out extreme values in `Age` and `Fare`

SECTION 9. Export Cleaned Data
- Saved the final cleaned dataset as `cleaned_titanic.csv`

## 🗃️ Output
- Final Cleaned CSV: `cleaned_titanic.csv`
- Contains no missing values, all features encoded/scaled, and outliers removed.

## 📸 Screenshots
![image](https://github.com/user-attachments/assets/11161ad9-01c2-4f97-8ec4-b9666ccdf49a)

=========================================================================   Task 2 - Exploratory Data Analysis (EDA)  ================================================================================

This repository showcases EDA on the Titanic dataset to explore patterns, relationships, and insights using visualization and statistical techniques.

## 🎯 Objective

To perform exploratory data analysis and visualize data distributions, correlations, and categorical relationships.

## 🧰 Tools & Libraries Used

- Python
- Pandas
- Seaborn
- Matplotlib
  
## 🛠️ Steps Performed - The index number is the section of the code

1. Imported Data
- Loaded the Titanic dataset from a public GitHub URL.

2. Summary Statistics
- Used `.describe()` and `.info()` to get initial insights into the dataset structure and missing values.

3. Distribution Analysis
- Plotted histograms and KDE plots for numerical features like `Age`.
- Boxplots for detecting skewed distributions and outliers in `Fare`.

4. Correlation Analysis
- Created a heatmap of numeric feature correlations using `.corr()` and `seaborn.heatmap()`.

5. Pairplot Analysis
- Used `pairplot` to analyze interactions between `Age`, `Fare`, `Pclass`, and `Survived`.

6. Categorical Insights
- Count plots to explore the relationship between `Sex` and `Survived`.

## 📸 Screenshots 

![image](https://github.com/user-attachments/assets/ae482910-ae36-422a-91ee-97cda43223ae)

![image](https://github.com/user-attachments/assets/54f750aa-b663-4640-90f5-c5e433d26ef7)

![image](https://github.com/user-attachments/assets/eb004567-0f3f-43d0-ab56-a017b585ca35)

![image](https://github.com/user-attachments/assets/4674462b-c8bf-442d-b54a-34d98f2faafb)

![image](https://github.com/user-attachments/assets/9d62ff03-7963-43d3-b96c-d2572f3bc242)

=========================================================================== Task 3: Linear Regression =========================================================================

# 📈 Task 3: Linear Regression – Titanic Dataset

This repository implements both **Simple Linear Regression** and **Multiple Linear Regression** models on the Titanic dataset, evaluating model performance and visualizing results.

---

## 🎯 Objective

- To implement and understand simple and multiple linear regression.
- To evaluate model performance using MAE, MSE, and R².
- To plot regression lines and interpret model coefficients.

---

## 🧰 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---
## 🛠️ Steps Performed - The index number is the section of the code 

1. Imported Data
- Loaded the cleaned Titanic dataset (`cleaned_titanic.csv`).

2. Preprocessing
- Checked for missing values and confirmed dataset readiness.

3. Simple Linear Regression (SLR)
- Selected `Age` as the single independent variable to predict `Fare`.
- Split the data into training and testing sets.
- Fitted a simple linear regression model.
- Plotted the **Regression Line** (Actual vs Predicted Fare based on Age).

4. Multiple Linear Regression (MLR)
- Selected `Age`, `Pclass`, and `FamilySize` as independent variables.
- Split the data into training and testing sets.
- Fitted a multiple linear regression model.
- Plotted **Actual vs Predicted Fares** (scatter plot).

5. Model Evaluation
- Calculated:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score** for both SLR and MLR.
- Printed the model coefficients to interpret feature impacts.

## 📊 Visualizations Included

- **Scatter plot + Regression Line** for Simple Linear Regression.
- **Scatter plot** of Actual vs Predicted values for Multiple Linear Regression.

----
## Screenshots

-> Simple Linear Regression
![image](https://github.com/user-attachments/assets/4569c4db-3659-4e51-a46a-4a00214faea2)

-> Multiple Regression 
![image](https://github.com/user-attachments/assets/2046e560-f1ad-486f-8354-30e014d89f80)


=========================================================================== Task 4: Classification with Logistic Regression  =========================================================================

# Task 4: Binary Classification with Logistic Regression

This task is all about building a binary classifier using Logistic Regression. I used the Breast Cancer Wisconsin dataset to classify tumors as malignant or benign and evaluated the model using real-world metrics like ROC-AUC, precision, recall, and more.

---

## 🧠 Objective

To train and evaluate a Logistic Regression model on a binary classification problem. Understand key evaluation metrics and how the sigmoid function shapes decision boundaries.

---

## 🛠️ Tools Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

---
## ⚙️ Steps Performed - The index number is the section of the code

### 1. Loaded the Dataset
- Used the Breast Cancer Wisconsin dataset from Kaggle.
- Cleaned it by removing unnecessary columns and encoded the target (`diagnosis`: M = 1, B = 0).

### 2. Preprocessed the Data
- Split the data into train/test sets (80/20).
- Standardized the features using `StandardScaler`.

### 3. Built the Model
- Trained a Logistic Regression model using `sklearn.linear_model`.
- Predicted outcomes and generated probabilities for ROC-AUC analysis.

### 4. Evaluated the Model
- Created a **confusion matrix** to visualize performance.
- Calculated **precision**, **recall**, and **classification report**.
- Plotted the **ROC Curve** and calculated **AUC score**.

### 5. Tuned the Threshold
- Adjusted the decision threshold to 0.4 (default is 0.5).
- Printed a new confusion matrix to show how predictions change.

### 6. Sigmoid Function
- Plotted the sigmoid curve to show how logistic regression maps linear output into probabilities between 0 and 1.
----

## Screenshots

-> Confusion matrix
![image](https://github.com/user-attachments/assets/d8332e75-a96b-46fb-a228-f62437aa0a1d)

-> ROC Curve
![image](https://github.com/user-attachments/assets/88468565-68ee-49d2-9376-539d6c954d3d)

->Sigmoid function
![image](https://github.com/user-attachments/assets/73f86e73-3932-48e6-b305-de1418f49899)


=========================================================================== Task 5: K-Means Clustering   =========================================================================

# Task 5: Heart Disease Prediction – Decision Tree & Random Forest Classifiers

📌 **Objective**  
The goal of this task is to build an ML model to predict heart disease using a variety of patient attributes.

🧰 **Tools & Libraries Used**  
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- scikit-learn  
---

Dataset Used - heart.csv
---

⚙️ **Workflow & Steps**

### **SECTION 1. Import Libraries**
All necessary libraries for data processing, visualization, and machine learning were imported, including `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

### **SECTION 2. Load & Preprocess Data**
- The dataset is loaded from an online URL using `pandas.read_csv()`. 
- The feature columns (`X`) and target column (`y`) are separated for model training.
- The data is split into training and testing sets using `train_test_split`.

### **SECTION 3. Decision Tree Classifier**
- A Decision Tree model with a maximum depth of 4 is trained on the dataset.
- The Decision Tree is visualized to understand the decision rules, with a plot displaying the splits and the target classes (`Disease` or `No Disease`).
- The model's performance is evaluated using the classification report and confusion matrix.

### **SECTION 4. Random Forest Classifier**
- A Random Forest model with 100 estimators is trained on the same dataset.
- The performance of the Random Forest model is evaluated using the classification report.
- Feature importance is computed and visualized to show the most significant features for predicting heart disease.

### **SECTION 5. Cross-Validation**
- The Random Forest model is further evaluated using 5-fold cross-validation to check its performance on unseen data.

🗃️ **Output**
- **Classification Report and Confusion Matrix**: These metrics are provided for both the Decision Tree and Random Forest models, showing precision, recall, and F1-score.
- **Feature Importance Plot**: A bar plot showing the top 10 features contributing the most to the Random Forest model's predictions.

📸 **Screenshots**

![image](https://github.com/user-attachments/assets/60788b50-b883-4032-a8ad-2c642d8e81d3)

![image](https://github.com/user-attachments/assets/c21d2786-3b25-41d6-8026-b09b3f345dc1)

![image](https://github.com/user-attachments/assets/158a5cda-b433-438a-b2e6-78228e7bdca2)

=========================================================================== Task 6: Classification using KNN   =========================================================================

# 🎯 Task 6: Classification with K-Nearest Neighbors (KNN)

In this task, I explored the K-Nearest Neighbors (KNN) algorithm on the famous Iris dataset. The goal was to classify flower species based on their features and evaluate model performance using multiple K values and visualizations.
---

## 🧠 Objective

To implement KNN for multi-class classification and understand how changing the number of neighbors (`k`) affects accuracy and decision boundaries.
---

## 🛠️ Tools Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
---
## Dataset used - Iris
---
## ⚙️ What I Did

### 1. Loaded the Dataset
- Used the built-in **Iris** dataset from `sklearn.datasets`.
- Converted it into a pandas DataFrame for better handling and readability.

### 2. Preprocessed the Data
- Split the dataset into **train** and **test** sets using `train_test_split` (80/20).
- Standardized features using `StandardScaler` to improve distance-based classification.

### 3. Built the Model
- Used `KNeighborsClassifier` from `sklearn.neighbors`.
- Trained the model for different values of `k` (e.g. 1 to 20).
- Recorded and visualized accuracy scores for each `k`.

### 4. Evaluated the Model
- Calculated accuracy, confusion matrix, and classification report for the best `k`.
- Used **pairplots** to visualize separation between different classes.
- Plotted a graph showing how accuracy varies with different `k` values.

### 5. Visualized Decision Boundaries
- Plotted the **decision boundaries** using two selected features (for 2D visualization) to intuitively understand how KNN separates the classes.

---
### Screenshots

![image](https://github.com/user-attachments/assets/e66dfa6e-00e7-42a2-a4da-6ed6c8b7e95b)

![image](https://github.com/user-attachments/assets/0152f708-0c9e-4096-8fa9-e2c48cbb8c7f)

![image](https://github.com/user-attachments/assets/8bc50d5b-8f06-49b1-b0a8-99e56b6582ce)

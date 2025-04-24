========================================================================= Task 1: Data Cleaning & Preprocessing – Titanic Dataset ===================================================================================

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

=========================================================================                                                         ===================================================================================

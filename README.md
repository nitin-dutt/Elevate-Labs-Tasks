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

=========================================================================   Task 2 - Exploratory Data Analysis (EDA)  ===================================================================================

This repository showcases EDA on the Titanic dataset to explore patterns, relationships, and insights using visualization and statistical techniques.

## 🎯 Objective

To perform exploratory data analysis and visualize data distributions, correlations, and categorical relationships.

## 🧰 Tools & Libraries Used

- Python
- Pandas
- Seaborn
- Matplotlib
  
## 🛠️ Steps Performed

### 1. Imported Data
- Loaded the Titanic dataset from a public GitHub URL.

### 2. Summary Statistics
- Used `.describe()` and `.info()` to get initial insights into the dataset structure and missing values.

### 3. Distribution Analysis
- Plotted histograms and KDE plots for numerical features like `Age`.
- Boxplots for detecting skewed distributions and outliers in `Fare`.

### 4. Correlation Analysis
- Created a heatmap of numeric feature correlations using `.corr()` and `seaborn.heatmap()`.

### 5. Pairplot Analysis
- Used `pairplot` to analyze interactions between `Age`, `Fare`, `Pclass`, and `Survived`.

### 6. Categorical Insights
- Count plots to explore the relationship between `Sex` and `Survived`.

## 📸 Screenshots (Optional)

![image](https://github.com/user-attachments/assets/ae482910-ae36-422a-91ee-97cda43223ae)

![image](https://github.com/user-attachments/assets/54f750aa-b663-4640-90f5-c5e433d26ef7)

![image](https://github.com/user-attachments/assets/eb004567-0f3f-43d0-ab56-a017b585ca35)

![image](https://github.com/user-attachments/assets/4674462b-c8bf-442d-b54a-34d98f2faafb)

![image](https://github.com/user-attachments/assets/9d62ff03-7963-43d3-b96c-d2572f3bc242)


---

## 🧾 Output

No CSV is generated in this task — all insights are visualized and explored interactively.

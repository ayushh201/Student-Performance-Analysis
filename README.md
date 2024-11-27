# Student-Performance-Analysis

## 1. Motivation Behind the Project
The motivation behind the Student Performance Analysis project is to leverage data science and machine learning techniques to analyze and predict student performance based on various parameters. Education is a crucial area, and understanding factors that influence student performance can help improve teaching methods, curriculum design, and provide valuable insights for educational institutions. By using machine learning algorithms, we aim to build a predictive model that can forecast student performance, helping educators identify struggling students and take proactive steps to improve their learning outcomes.

## 2. Type of Project
This project falls under the category of Predictive Modeling and Data Analysis. The goal is to train multiple machine learning models to predict students' academic performance based on historical data. The project also includes Exploratory Data Analysis (EDA) to analyze and understand the dataset, visualize key relationships, and perform feature selection to improve model performance.

## 3. Critical Analysis
The key challenge in this project lies in ensuring the quality and cleanliness of the dataset. Missing values, outliers, and irrelevant features could drastically affect the performance of the models. By conducting thorough exploratory data analysis (EDA), we can detect these issues and handle them appropriately. Furthermore, selecting the right features that impact student performance is crucial for model accuracy. Given the complexity of student performance, incorporating various machine learning algorithms is essential to test different approaches and determine the best-suited model for this problem. A critical analysis of model performance metrics like RMS error, Mean Squared Error (MSE), and R² score will be performed to ensure the model's predictive capability.

## 4. Programming Language
The programming language used for this project is Python, which is widely recognized for its versatility in data analysis and machine learning. Python offers a wide range of libraries that support data manipulation, machine learning, and visualization, making it an ideal choice for this project.

## 5. Tools & Python Libraries Used
The following tools and Python libraries were utilized for the project:
Pandas: For data manipulation, cleaning, and analysis.
NumPy: For handling arrays and mathematical operations.
Matplotlib & Seaborn: For data visualization and plotting graphs.
Scikit-learn: For machine learning models and evaluation metrics.
XGBoost: For implementing the XGBRegressor model.
CatBoost: For implementing the CatBoost model.
Statsmodels: For statistical analysis and model evaluation.
Jupyter Notebook: For interactive coding, analysis, and visualization.

## 6. Software
The primary software used for the project is Python 3, with all dependencies installed in a virtual environment. The development and analysis were conducted on Jupyter Notebooks to facilitate interactive coding and visualization. Anaconda was used to manage the Python environment and libraries, providing a streamlined setup for the project.

## 7. Methodology Summary
The methodology of the project can be summarized in the following steps:
Data Collection: The dataset is collected, which typically includes student attributes such as parental level of education, gender, race ethnicity, test preparation, reading-writing scores, etc.

Exploratory Data Analysis (EDA): A thorough analysis is performed to identify patterns, correlations, and relationships between features and the target variable. 

Data Preprocessing: This includes handling missing values, encoding categorical variables, scaling numerical values, and splitting the dataset into training and testing sets.

Model Training: Multiple machine learning models are trained on the dataset, including Linear Regression, Ridge Regression, Random Forest, CatBoost, AdaBoost, XGBRegressor, Lasso, K-Nearest Neighbors, and Decision Tree.

Model Evaluation: The models are evaluated using metrics such as Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R² score for both the training and testing sets.

Model Selection: Based on the performance metrics, the best-performing model is selected for prediction and analysis.

Results Interpretation: The results are analyzed to understand the key factors influencing student performance, and the insights are used for recommendations.

## 8. Machine Learning Algorithms Applied
The following machine learning algorithms were applied in the project:

Linear Regression: A simple approach to model the relationship between the dependent and independent variables.

Ridge Regression: A variation of linear regression that includes regularization to reduce overfitting.

Random Forest: An ensemble method using multiple decision trees to improve accuracy and prevent overfitting.

CatBoosting: A gradient boosting algorithm optimized for categorical features.

AdaBoost: An ensemble technique that adjusts the weights of weak learners to improve performance.

XGBRegressor: A powerful boosting algorithm that is highly effective in regression tasks.

Lasso Regression: A regression technique that uses L1 regularization to select features.

K-Nearest Neighbors (KNN): A non-parametric method that makes predictions based on the closest training examples in the feature space.

Decision Tree: A tree-like model used for classification and regression tasks.

## 9. Model Evaluation Table

| Model                  | RMSE (Training Set) | RMSE (Test Set) | MAE (Training Set) | MAE (Test Set) | R² Score (Training Set) | R² Score (Test Set) |
|------------------------|---------------------|-----------------|--------------------|----------------|-------------------------|---------------------|
| Linear Regression      | 5.3402             | 5.4214          | 4.2723            | 4.2253         | 0.8735                 | 0.8792             |
| Lasso                  | 6.5938             | 6.5197          | 5.2063            | 5.1579         | 0.8071                 | 0.8253             |
| Ridge                  | 5.3233             | 5.3904          | 4.2650            | 4.2111         | 0.8743                 | 0.8806             |
| K-Neighbors Regressor  | 5.7091             | 7.2583          | 4.5175            | 5.6370         | 0.8554                 | 0.7835             |
| Decision Tree          | 0.2795             | 7.6704          | 0.0187            | 6.0750         | 0.9997                 | 0.7582             |
| Random Forest Regressor| 2.3019             | 5.9305          | 1.8292            | 4.6165         | 0.9765                 | 0.8555             |
| XGBRegressor           | 1.0073             | 6.4733          | 0.6875            | 5.0577         | 0.9955                 | 0.8278             |
| CatBoosting Regressor  | 3.0427             | 6.0086          | 2.4054            | 4.6125         | 0.9589                 | 0.8516             |
| AdaBoost Regressor     | 5.8179             | 6.0791          | 4.7580            | 4.6871         | 0.8499                 | 0.8481             |

The Random Forest, Linear Regression and Decision Tree models performed exceptionally well, with lower RMSE and higher R² scores compared to other models.

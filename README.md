# Predictive-Modeling-for-Customer-Churn
Customer churn prediction is a widely applicable problem for many industries, especially in sectors like telecommunications, SaaS, and banking. The idea is to predict whether a customer will leave a company (churn) based on historical customer data.
# Customer Churn Prediction

This project aims to predict whether a customer will leave a company (churn) based on historical customer data. The dataset contains various features like customer demographics, usage patterns, and account information. The goal is to build a machine learning model that can predict customer churn with high accuracy.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment) (Optional)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description
Customer churn prediction is critical for many industries, especially in sectors like telecommunications, banking, and SaaS, where retaining customers is key to sustaining business growth. In this project, I used a publicly available dataset to predict if a customer will churn based on various features such as:
- Demographic information (age, gender, income)
- Account information (tenure, contract type, payment method)
- Usage patterns (monthly charges, total charges)

The project involves data cleaning, exploratory data analysis (EDA), feature engineering, and training machine learning models.

## Dataset
The dataset used in this project is the **Telco Customer Churn Dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

The dataset contains the following features:
- **customerID**: Unique ID for each customer
- **gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen
- **Partner**: Whether the customer has a partner
- **Dependents**: Whether the customer has dependents
- **tenure**: Number of months the customer has been with the company
- **PhoneService**: Whether the customer has phone service
- **MultipleLines**: Whether the customer has multiple lines
- **InternetService**: Type of internet service the customer has
- **OnlineSecurity**: Whether the customer has online security
- **TechSupport**: Whether the customer has tech support
- **StreamingTV**: Whether the customer has streaming TV
- **StreamingMovies**: Whether the customer has streaming movies
- **Contract**: Type of contract the customer has
- **PaperlessBilling**: Whether the customer has paperless billing
- **PaymentMethod**: Payment method used by the customer
- **MonthlyCharges**: Monthly charges the customer is paying
- **TotalCharges**: Total charges the customer has paid
- **Churn**: Whether the customer churned (target variable)

## Data Preprocessing
- Handled missing values by imputing or dropping columns as necessary.
- Encoded categorical variables using One-Hot Encoding and Label Encoding.
- Scaled numerical features to standardize the input data.
- Split the dataset into training and test sets (80/20 split).

## Exploratory Data Analysis (EDA)
The EDA phase included:
- Visualizing the distribution of numerical variables using histograms and box plots.
- Investigating correlations between features and the target variable.
- Analyzing categorical variables and their impact on churn rate.
- Generating summary statistics to better understand the data.

## Model Building
I tried several machine learning models to predict customer churn, including:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Support Vector Machine (SVM)**

Each model was evaluated using cross-validation and hyperparameter tuning to find the best model for the task.

## Model Evaluation
The models were evaluated using:
- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

The **Random Forest Classifier** performed the best with an **accuracy of 85%** and an **AUC score of 0.90**.

## Deployment (Optional)
For those interested in deploying the model, the application can be served through a simple web interface using **Flask** or **Streamlit**. The user can input customer information and receive a prediction on whether the customer is likely to churn.

- [Streamlit Demo Link](#) *(optional if deployed)*

## Technologies Used
- **Python**: Main programming language
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models and evaluation metrics
- **matplotlib** & **seaborn**: Data visualization
- **XGBoost**: Advanced model for classification
- **Flask** (Optional): Web framework for deployment
- **Streamlit** (Optional): Interactive web application for model deployment

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Hibamessai/Predictive-Modeling-for-Customer-Churn.git
    cd churn-prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script to train the models.

## Usage
After setting up the project, you can:
- Explore the **notebooks/** directory to see the code for EDA, preprocessing, and model training.
- Run `churn_prediction.py` to train the models and make predictions.
- Use the `deploy.py` (if available) to run the Flask app or Streamlit app for model deployment.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


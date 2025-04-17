import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
import joblib

# Set the title of the app
st.title("AI-Powered Automated Data Insights Platform")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Proceed if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check if the dataset is empty
    if df.empty:
        st.error("The uploaded dataset is empty.")
    else:
        st.write("### Dataset Preview")
        st.write(df.head())

        st.write("### Basic Dataset Information")
        st.write(df.describe())

        st.write("### Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write(missing_values[missing_values > 0])
        else:
            st.write("No missing values found in the dataset.")
        
        # Missing data heatmap
        st.write("### Missing Data Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        st.pyplot(fig)
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, drop_first=True)

        st.write("### Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)

        # Generate Sweetviz EDA report
        if st.button("Generate Automated EDA Report"):
            report = sv.analyze(df)
            report_path = "sweetviz_report.html"
            report.show_html(report_path, open_browser=False)
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download EDA Report", 
                    data=file, 
                    file_name="EDA_Report.html",
                    mime="text/html"
                )

        st.write("### Automated Machine Learning")
        
        # Target column selection
        target_column = st.selectbox("Select Target Column for Prediction", df.columns)
        
        if target_column:
            # Prepare features and target variable
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Drop irrelevant columns
            drop_cols = ['Name', 'Ticket', 'Cabin']
            X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')

            # One-hot encode categorical columns
            X = pd.get_dummies(X, drop_first=True)

            # Handle non-numeric target variable
            if not is_numeric_dtype(y):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            # Impute missing values
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Choose model based on target variable type
            if is_numeric_dtype(y):
                model = RandomForestRegressor()
            else:
                model = RandomForestClassifier()

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate the model
            if is_numeric_dtype(y):
                error = mean_absolute_error(y_test, y_pred)
                st.write(f"Mean Absolute Error: {error:.2f}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Save the trained model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "automl_model.pkl")
            joblib.dump(model, model_path)
            st.success("AutoML Model Trained & Saved!")

            # Allow the user to download the trained model
            with open(model_path, "rb") as file:
                st.download_button("Download Trained Model", file, "automl_model.pkl")

# Footer
st.write("### Built with ❤ by Akash")

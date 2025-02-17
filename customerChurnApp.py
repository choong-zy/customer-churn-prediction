import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
loaded_model = joblib.load('C:/Users/choong zhi yang/Downloads/trained_model (4).sav')

# Function to preprocess input data
# Function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to binary indicators
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    data['SeniorCitizen'] = data['SeniorCitizen'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['Partner'] = data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['Dependents'] = data['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['PhoneService'] = data['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['PaperlessBilling'] = data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Handle multiple lines
    data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})

    # Handle internet service
    data['InternetService_DSL'] = data['InternetService'].apply(lambda x: 1 if x == 'DSL' else 0)
    data['InternetService_Fiber optic'] = data['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
    data['InternetService_No'] = data['InternetService'].apply(lambda x: 1 if x == 'No' else 0)

    data['OnlineSecurity'] = data['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    data['OnlineBackup'] = data['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    data['DeviceProtection'] = data['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    data['TechSupport'] = data['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    data['StreamingTV'] = data['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    data['StreamingMovies'] = data['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})

    # Handle contract
    data['Contract_Month-to-month'] = data['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
    data['Contract_One year'] = data['Contract'].apply(lambda x: 1 if x == 'One year' else 0)
    data['Contract_Two year'] = data['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)

    # Handle payment method
    data['PaymentMethod_Bank transfer (automatic)'] = data['PaymentMethod'].apply(lambda x: 1 if x == 'Bank transfer (automatic)' else 0)
    data['PaymentMethod_Credit card (automatic)'] = data['PaymentMethod'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)
    data['PaymentMethod_Electronic check'] = data['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
    data['PaymentMethod_Mailed check'] = data['PaymentMethod'].apply(lambda x: 1 if x == 'Mailed check' else 0)

    # Drop original categorical columns
    data.drop(['InternetService', 'Contract', 'PaymentMethod'], axis=1, inplace=True)

    scaler = MinMaxScaler()

    # Scale numeric features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data


# Function to make predictions
def predict_churn(input_data):
    preprocessed_input = preprocess_input(input_data)
    prediction = loaded_model.predict(preprocessed_input)
    return prediction[0]

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # Get input from the user
    
    senior_citizen = st.checkbox("Senior Citizen")
    partner = st.checkbox("Has a Partner")
    dependents = st.checkbox("Has Dependents")
    phone_service = st.checkbox("Phone Service")
    multiple_lines = st.checkbox("Multiple Lines")
    online_security = st.checkbox("Online Security")
    online_backup = st.checkbox("Online Backup")
    device_protection = st.checkbox("Device Protection")
    tech_support = st.checkbox("Tech Support")
    streaming_tv = st.checkbox("Streaming TV")
    streaming_movies = st.checkbox("Streaming Movies")
    paperless_billing = st.checkbox("Paperless Billing")
    gender = st.radio("Gender", ('Male', 'Female'))
    internet_service = st.radio("Internet Service", ('DSL', 'Fiber optic', 'No'))
    contract = st.radio("Contract", ('Month-to-month', 'One year', 'Two year'))
    payment_method = st.radio("Payment Method", ('Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'))
    tenure = st.number_input("Tenure (in months)")
    monthly_charges = st.number_input("Monthly Charges")
    total_charges = st.number_input("Total Charges")
    
    # Create a dictionary with the input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': 'Yes' if senior_citizen else 'No',
        'Partner': 'Yes' if partner else 'No',
        'Dependents': 'Yes' if dependents else 'No',
        'tenure': tenure,
        'PhoneService': 'Yes' if phone_service else 'No',
        'MultipleLines': 'Yes' if multiple_lines else 'No',
        'InternetService': internet_service,
        'OnlineSecurity': 'Yes' if online_security else 'No',
        'OnlineBackup': 'Yes' if online_backup else 'No',
        'DeviceProtection': 'Yes' if device_protection else 'No',
        'TechSupport': 'Yes' if tech_support else 'No',
        'StreamingTV': 'Yes' if streaming_tv else 'No',
        'StreamingMovies': 'Yes' if streaming_movies else 'No',
        'Contract': contract,
        'PaperlessBilling': 'Yes' if paperless_billing else 'No',
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = predict_churn(input_df)
        if prediction == 1:
            st.write("The customer is predicted to churn.")
        else:
            st.write("The customer is predicted not to churn.")

if __name__ == "__main__":
    main()

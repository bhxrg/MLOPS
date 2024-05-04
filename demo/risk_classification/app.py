import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-fitted pipeline and model
def load_artifact(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Streamlit app definition
def main():
    st.title('Credit Risk Prediction App')

    # Load the pre-fitted data processing pipeline and model
    pipeline = load_artifact('data_processing_pipeline.pkl')
    model = load_artifact('best_classifier.pkl')
    label_encoder = load_artifact('label_encoder.pkl')

    # User input fields
    st.header("Enter customer details:")
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Income", min_value=0)
    employment_type = st.selectbox("Employment Type", ['Salaried', 'Unemployed', 'Self-employed'])
    residence_type = st.selectbox("Residence Type", ['Parental Home', 'Rented', 'Owned'])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term", min_value=0)
    previous_default = st.selectbox("Previous Default", ['Yes', 'No'])

    # Predict button
    if st.button('Predict Risk Category'):
        # Assemble user inputs into a dataframe matching the expected model input structure
        input_df = pd.DataFrame([[age, income, employment_type, residence_type, credit_score, loan_amount, loan_term, previous_default]],
                                columns=['Age', 'Income', 'EmploymentType', 'ResidenceType', 'CreditScore', 'LoanAmount', 'LoanTerm', 'PreviousDefault'])

        # Use the pre-fitted pipeline to transform the input data
        transformed_input = pipeline.transform(input_df)

        # Make prediction with the preloaded model
        prediction = model.predict(transformed_input)
        
        # Decode the prediction using the loaded label encoder
        decoded_prediction = label_encoder.inverse_transform(prediction)

        # Display the prediction
        st.subheader('Predicted Risk Category:')
        st.write(decoded_prediction[0])

if __name__ == '__main__':
    main()

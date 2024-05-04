import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename='logfile_UI.txt',  # Specify the file name
    level=logging.DEBUG,      # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date/time format
)

# Load the pre-fitted pipeline and model


def load_artifact(filename):
    """
    Loads a pickled artifact from the specified filename.

    Args:
        filename (str): Path to the pickled artifact file.

    Returns:
        object: The loaded artifact (can be data or a model).

    Raises:
        FileNotFoundError: If the artifact file is not found at the specified path.
    """

    try:        
        with open(filename, 'rb') as file:        
            return pickle.load(file)

    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        raise  # Re-raise the exception for handling in the calling code


def main():
    st.title('Credit Risk Prediction App')

    # Load the pre-fitted data processing pipeline and model
    pipeline = load_artifact('data_processing_pipeline.pkl')
    logging.info(f"Pipeline loaded successfully")
    model = load_artifact('best_classifier.pkl')
    logging.info(f"Model loaded successfully")
    label_encoder = load_artifact('label_encoder.pkl')
    logging.info(f"Label Encoder loaded successfully")

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
        logging.info(f"User input data frame created")
        # Use the pre-fitted pipeline to transform the input data
        transformed_input = pipeline.transform(input_df)
        logging.info(f"User input data frame is transformed")
        # Make prediction with the preloaded model
        prediction = model.predict(transformed_input)
        logging.info(f"Received Predicion: {prediction}")

        # Decode the prediction using the loaded label encoder
        decoded_prediction = label_encoder.inverse_transform(prediction)
        logging.info(f"Predicion Decoded: {decoded_prediction}")
        # Display the prediction
        st.subheader('Predicted Risk Category:')
        st.write(decoded_prediction[0])

if __name__ == '__main__':
    logging.info(f"Calling UI main")
    main()

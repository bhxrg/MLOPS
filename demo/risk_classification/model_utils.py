
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost model and evaluates it using the test data.
    Outputs the confusion matrix and classification report.
    """
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_val)

    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train_encoded)

    # Predictions
    y_pred = model.predict(X_val)

    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    cr = classification_report(y_test_encoded, y_pred, target_names=le.classes_)
    print("Classification Report:")
    print(cr)

    # Save the model to disk
    filename = 'finalized_model.sav'
    label_encoder_filename = 'label_encoder.pkl'
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(le, open(label_encoder_filename, 'wb'))
    print(f"Model saved to {filename}")

    return model



def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_preprocessor(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def make_prediction(input_data):
    model_file = "C:/Users/bpanda31/Downloads/DSC/MLOPs/demo/finalized_model.sav"
    preprocessor_file = "C:/Users/bpanda31/Downloads/DSC/MLOPs/demo/data_processing_pipeline.pkl"
    label_encoder_file = "label_encoder.pkl"

    # Load the preprocessor and model
    model = load_model(model_file)
    preprocessor = load_preprocessor(preprocessor_file)
    le = pickle.load(open(label_encoder_file, 'rb'))

    # Renaming keys to match the expected column names used during model training
    input_data = {k.capitalize(): v for k, v in input_data.items()}
    input_data['EmploymentType'] = input_data.pop('Employmenttype')
    input_data['ResidenceType'] = input_data.pop('Residencetype')
    input_data['CreditScore'] = input_data.pop('Creditscore')
    input_data['LoanAmount'] = input_data.pop('Loanamount')
    input_data['LoanTerm'] = input_data.pop('Loanterm')
    input_data['PreviousDefault'] = input_data.pop('Previousdefault')

    # Create DataFrame for prediction
    df = pd.DataFrame([input_data])

    # Debug: Print the DataFrame to ensure it's correctly formatted
    print("DataFrame for prediction:", df)

    processed_data = preprocessor.transform(df)
    prediction_result_num = model.predict(processed_data)

    print('prediction_result::',prediction_result_num)
    prediction_labels = le.inverse_transform(prediction_result_num)
    print('prediction_result::',prediction_labels)
    return prediction_labels

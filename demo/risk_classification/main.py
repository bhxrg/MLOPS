import pandas as pd
from data_preprocessing import create_data_pipeline, save_pipeline, load_pipeline, split_data, encode_response_variable
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices


def main():
    
    df = pd.read_csv('C:/Users/bpanda31/Downloads/DSC/MLOPs/demo_streamlit/risk_classification/data/Banking_Credit_Risk_Data.csv')
    X = df.drop(['CustomerID', 'RiskCategory'], axis=1)
    y = df['RiskCategory']

    y_encoded = encode_response_variable(y)
    # Create the pipeline
    pipeline = create_data_pipeline()
    pipeline.fit(X)

    # Save the pipeline for later use during predictions
    save_pipeline(pipeline, 'data_processing_pipeline.pkl')

    # Transform the data using the fit_transform method
    X_transformed = pipeline.fit_transform(X)

    # Split the data
    X_train, X_val, y_train, y_val = split_data(X_transformed, y_encoded)
    # print('train_labels', y_train)

    best_model = training_pipeline(X_train, y_train)
    # print(best_model)

    predictions = prediction_pipeline(X_val)
    # print(predictions)

    conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val)
    # Optionally print or log the results
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy Score:", acc_score)
    print("Classification Report:\n", class_report)

if __name__ == "__main__":
    main()
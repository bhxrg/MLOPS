import pandas as pd
from flask import Flask

from app.model_utils import train_and_evaluate_model, load_model
from app.routes import main
from data_engineering.helper_functions import create_data_pipeline, save_pipeline, split_data

def main_process():
    data_path = "...data/Banking_Credit_Risk_Data.csv"
    df = pd.read_csv(data_path)

    # Separating features and target
    X = df.drop(['CustomerID', 'RiskCategory'], axis=1)
    y = df['RiskCategory']

    # Create the pipeline
    pipeline = create_data_pipeline()
    pipeline.fit(X)

    # Save the pipeline for later use during predictions
    save_pipeline(pipeline, 'data_processing_pipeline.pkl')

    # Transform the data using the fit_transform method
    X_transformed = pipeline.fit_transform(X)

    # Split the data
    X_train, X_val, y_train, y_val = split_data(X_transformed, y)

    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, y_train, X_val, y_val)
    
    # Save the trained model
    save_pipeline(model, 'finalized_model.sav')  # Ensure you use save_pipeline or similar method suitable for models

    print(X_train)  # Optional: print the training data to confirm
    print(model)    # Optional: print the model summary or evaluation metrics

# Set up Flask
app = Flask(__name__)
app.register_blueprint(main)

if __name__ == '__main__':
    main_process()  # Call main processing function which handles data loading, training etc.
    app.run(debug=True)  # Start the Flask application

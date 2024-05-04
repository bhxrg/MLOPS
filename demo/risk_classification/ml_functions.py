import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd


def training_pipeline(X_train, y_train):
    # Initialize the XGBoost classifier
    classification_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss'  # This is often used for multi-class classification
    )    
    # Train the model   
    classification_model.fit(X_train, y_train)
    print("Model trained successfully.")    
    with open('best_classifier.pkl', 'wb') as f:
        pickle.dump(classification_model, f)
        print("Model successfully pickled.")   
    return classification_model

articact_path = "C:/Users/bpanda31/Downloads/DSC/MLOPs/demo_streamlit/risk_classification/scripts/best_classifier.pkl"
def load_model(path):
    """ Load a pickled model from the specified path. """
    with open(articact_path, 'rb') as file:
        model = pickle.load(file)
    return model


def prediction_pipeline(X_val):    
    # Load the model    
    model = load_model('best_classifier.pkl')    
    predictions = model.predict(X_val)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder.inverse_transform(predictions)


def evaluation_matrices(X_val, y_val):  
    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Predictions assuming prediction_pipeline returns categorical labels
    pred_vals = prediction_pipeline(X_val)    
    # Decode y_val using the loaded LabelEncoder
    decoded_y_vals = label_encoder.inverse_transform(y_val)    
    # Calculate the confusion matrix with actual decoded labels
    conf_matrix = confusion_matrix(decoded_y_vals, pred_vals, labels=label_encoder.classes_)    
    # Additional evaluation metrics
    acc_score = accuracy_score(decoded_y_vals, pred_vals)
    class_report = classification_report(decoded_y_vals, pred_vals)    
    # Print the confusion matrix with class labels for better understanding
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))
    print("\nAccuracy Score:", acc_score)
    print("\nClassification Report:\n", class_report)
    return conf_matrix, acc_score, class_report


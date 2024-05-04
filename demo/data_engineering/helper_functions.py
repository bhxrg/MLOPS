from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

def create_data_pipeline():
    """
    Creates a data processing pipeline for both categorical and numerical features.
    This pipeline includes OneHotEncoding for categorical features with the first category dropped
    and MinMax scaling for numerical features.
    """
    categorical_features = ['EmploymentType', 'ResidenceType', 'PreviousDefault']
    numerical_features = ['Age', 'Income', 'CreditScore', 'LoanAmount', 'LoanTerm']

    # Setting up the transformations for categorical and numerical data
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ])

    # Creating a pipeline that applies the column transformer
    pipeline = Pipeline([
        ('col_transformer', column_transformer)
    ])

    return pipeline

def save_pipeline(pipeline, filename):
    """
    Saves the pipeline to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)

def load_pipeline(filename):
    """
    Loads a pipeline from a file.
    """
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

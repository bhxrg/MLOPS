from flask import Blueprint, render_template, request
from .model_utils import  make_prediction  # Import with alias

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction='',data ={})

@main.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        data = request.form.to_dict()
        print("Form Data Received:", data)
        try:
            prediction = make_prediction(data)
            print('prediction::::',prediction)
        except Exception as e:
            print("Error during prediction:", e) 
            prediction = f"Error: {e}"
            # To display the error in the template if needed
        return render_template('index.html', prediction=prediction, data = data)
    return render_template('index.html', prediction="", data={})

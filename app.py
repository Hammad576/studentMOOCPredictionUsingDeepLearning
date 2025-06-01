from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the SimpleNN model
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        logging.debug(f"Input shape to forward: {x.shape}")
        x = self.model(x)
        logging.debug(f"Output shape from forward: {x.shape}")
        return x

# Load model, scaler, and label encoder
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(input_size=16, num_classes=4).to(device)
    model.load_state_dict(torch.load('model/model.pth', map_location=device))
    model.eval()
    scaler = joblib.load('model/feature_scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    logging.info("Model, scaler, and label encoder loaded successfully")
except Exception as e:
    logging.error(f"Error loading model, scaler, or label encoder: {e}", exc_info=True)
    raise

FEATURES = [
    'studied_credits', 'forumng', 'oucontent', 'resource',
    'highest_education_A Level or Equivalent', 'highest_education_HE Qualification',
    'imd_band_0-10%', 'imd_band_90-100%', 'age_band_0-35', 'disability_Y',
    'homepage', 'subpage', 'gender_M', 'code_module_AAA', 'date',
    'highest_education_Lower Than A Level'
]
NUMERICAL_FEATURES = ['studied_credits', 'forumng', 'oucontent', 'resource', 'homepage', 'subpage', 'date']
RANGES = {
    'studied_credits': (30, 420),
    'forumng': (0, 107),
    'oucontent': (0, 344),
    'resource': (0, 19),
    'highest_education_A Level or Equivalent': (0, 1),
    'highest_education_HE Qualification': (0, 1),
    'imd_band_0-10%': (0, 1),
    'imd_band_90-100%': (0, 1),
    'age_band_0-35': (0, 1),
    'disability_Y': (0, 1),
    'homepage': (0, 65),
    'subpage': (0, 22),
    'gender_M': (0, 1),
    'code_module_AAA': (0, 1),
    'date': (0, 269),
    'highest_education_Lower Than A Level': (0, 1)
}

def load_dataset():
    try:
        df = pd.read_csv('data/reduced_dataset.csv')
        logging.info("Dataset loaded successfully")
        
        feature_means = df[['studied_credits', 'forumng', 'oucontent', 'resource', 'homepage', 'subpage', 'date']].mean().to_dict()
        
        outcome_counts = df['final_result'].value_counts().to_dict()
        
        gender_counts = {
            'female_pass': len(df[(df['gender_F'] == 1) & (df['final_result'] == 'Pass')]),
            'female_fail': len(df[(df['gender_F'] == 1) & (df['final_result'] == 'Fail')]),
            'female_distinction': len(df[(df['gender_F'] == 1) & (df['final_result'] == 'Distinction')]),
            'female_withdrawn': len(df[(df['gender_F'] == 1) & (df['final_result'] == 'Withdrawn')]),
            'male_pass': len(df[(df['gender_M'] == 1) & (df['final_result'] == 'Pass')]),
            'male_fail': len(df[(df['gender_M'] == 1) & (df['final_result'] == 'Fail')]),
            'male_distinction': len(df[(df['gender_M'] == 1) & (df['final_result'] == 'Distinction')]),
            'male_withdrawn': len(df[(df['gender_M'] == 1) & (df['final_result'] == 'Withdrawn')])
        }

        return {
            'feature_means': feature_means,
            'outcome_counts': outcome_counts,
            'gender_counts': gender_counts
        }
        
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return {
            'feature_means': {
                'studied_credits': 60.0,
                'forumng': 20.0,
                'oucontent': 50.0,
                'resource': 10.0,
                'homepage': 5.0,
                'subpage': 3.0,
                'date': 100.0
            },
            'outcome_counts': {
                'Fail': 100,
                'Pass': 300,
                'Withdrawn': 150,
                'Distinction': 50
            },
            'gender_counts': {
                'female_pass': 150,
                'female_fail': 50,
                'female_distinction': 25,
                'female_withdrawn': 75,
                'male_pass': 150,
                'male_fail': 50,
                'male_distinction': 25,
                'male_withdrawn': 75
            }
        }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def serve_data():
    return jsonify(load_dataset())

@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'Unknown')
    confidence = request.args.get('confidence', '0.85')
    
    advice_mapping = {
        'Pass': {
            'title': 'Great Job! Keep it Up',
            'tips': [
                'Maintain your current study routine.',
                'Continue active participation in forums.',
                'Complete quizzes and assignments on time.',
                'Review course materials regularly.'
            ]
        },
        'Fail': {
            'title': 'Improvement Suggestions',
            'tips': [
                'Increase dedicated study time.',
                'Engage more in forum discussions.',
                'Review and retake failed quizzes.',
                'Create a structured study schedule.'
            ]
        },
        'Withdrawn': {
            'title': 'Stay Engaged',
            'tips': [
                'Reconnect with course materials.',
                'Seek support from instructors or peers.',
                'Set achievable weekly goals.',
                'Use available resources actively.'
            ]
        },
        'Distinction': {
            'title': 'Outstanding Performance!',
            'tips': [
                'Share insights in forums to help peers.',
                'Explore advanced course topics.',
                'Maintain high engagement levels.',
                'Consider mentoring others.'
            ]
        }
    }
    
    return render_template(
        'result.html',
        prediction_result=prediction,
        confidence=float(confidence) * 100,
        advice=advice_mapping.get(prediction, {'title': 'Unknown Outcome', 'tips': []})
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = {}
        for feature in FEATURES:
            value = request.form.get(feature)
            if value is None or value == '':
                logging.error(f"Missing field: {feature}")
                return jsonify({'error': f'Missing field: {feature}'}), 400
            try:
                value = float(value)
            except ValueError:
                logging.error(f"Invalid value for {feature}: {value}")
                return jsonify({'error': f'Invalid value for {feature}: must be numeric'}), 400
            min_val, max_val = RANGES[feature]
            if not (min_val <= value <= max_val):
                logging.error(f"Value out of range for {feature}: {value}")
                return jsonify({'error': f'{feature} must be between {min_val} and {max_val}'}), 400
            inputs[feature] = value
        
        # Preprocess inputs
        df = pd.DataFrame([inputs])
        numerical = df[NUMERICAL_FEATURES].apply(lambda x: np.log1p(x - x.min()) if x.min() < 0 else np.log1p(x))
        numerical_scaled = scaler.transform(numerical)
        binary = df[[f for f in FEATURES if f not in NUMERICAL_FEATURES]].values
        inputs_scaled = np.hstack([numerical_scaled, binary]).astype(np.float32)
        inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(inputs_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            prediction = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        
        logging.info(f"Prediction: {prediction}, Confidence: {confidence.item()}")
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence.item()
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
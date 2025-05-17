from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define the CNN+LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTM(num_classes=4).to(device)
model.load_state_dict(torch.load('model/student_prediction_model.pth'))
model.eval()
scaler = joblib.load('model/scaler.pkl')

# Input feature names and ranges for validation
FEATURES = [
    'num_of_prev_attempts', 'studied_credits', 'forumng', 'oucontent', 'quiz', 'resource',
    'highest_education_a_level', 'highest_education_he', 'imd_band_0_10', 'imd_band_90_100',
    'age_band_0_35', 'disability_y'
]
RANGES = {
    'num_of_prev_attempts': (0, 5),
    'studied_credits': (30, 200),
    'forumng': (0, 100),
    'oucontent': (0, 200),
    'quiz': (0, 50),
    'resource': (0, 100),
    'highest_education_a_level': (0, 1),
    'highest_education_he': (0, 1),
    'imd_band_0_10': (0, 1),
    'imd_band_90_100': (0, 1),
    'age_band_0_35': (0, 1),
    'disability_y': (0, 1)
}

# Load dataset for graphs
def load_dataset():
    try:
        df = pd.read_csv('data/reduced_dataset.csv')
        
        # Calculate feature means
        feature_means = df[['num_of_prev_attempts', 'studied_credits', 'forumng', 
                           'oucontent', 'quiz', 'resource']].mean().to_dict()
        
        # Calculate outcome distribution
        outcome_counts = df['final_result'].value_counts().to_dict()
        
        # Calculate gender-based outcomes
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
        print(f"Error loading dataset: {e}")
        return {
            'feature_means': {
                'num_of_prev_attempts': 0.5,
                'studied_credits': 60.0,
                'forumng': 20.0,
                'oucontent': 50.0,
                'quiz': 10.0,
                'resource': 30.0
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
def model():
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
    # Add actual confidence calculation from your model
    confidence = round(np.random.uniform(75, 95), 1)  # Replace with real confidence
    
    advice_mapping = {
        'Pass': {
            'title': 'Great Job! Keep it Up',
            'tips': [
                'Maintain current study routine',
                'Continue forum participation',
                'Complete quizzes on time',
                'Regular material reviews'
            ]
        },
        'Fail': {
            'title': 'Improvement Suggestions',
            'tips': [
                'Increase study time',
                'Participate in forums',
                'Review failed quizzes',
                'Create study schedule'
            ]
        },
        # Add other mappings
    }
    
    return render_template('result.html',
                         prediction_result=prediction,
                         confidence=confidence,
                         advice=advice_mapping.get(prediction, {}))
                         
@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = []
        for feature in FEATURES:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing field: {feature}'}), 400
            try:
                value = float(value)
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}: must be numeric'}), 400
            min_val, max_val = RANGES[feature]
            if not (min_val <= value <= max_val):
                return jsonify({'error': f'{feature} must be between {min_val} and {max_val}'}), 400
            inputs.append(value)
        
        inputs = np.array(inputs, dtype=np.float32).reshape(1, -1)
        inputs = scaler.transform(inputs)
        inputs = torch.tensor(inputs.reshape(1, 1, 12), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            classes = ['Fail', 'Pass', 'Withdrawn', 'Distinction']
            prediction = classes[predicted.item()]
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
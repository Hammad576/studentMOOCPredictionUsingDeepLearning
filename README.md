# Student Outcome Prediction System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview
This system predicts student academic outcomes (Pass/Fail/Withdrawn/Distinction) using a hybrid CNN-LSTM deep learning model. The web application provides:
- Interactive outcome predictions
- Comprehensive data visualizations
- Gender-based performance analysis
- Course engagement metrics

Supervisor: Dr. Shehzad Rizwan  
Course: Database Lab  
Institution: Comsats University Islamabad Attock CAmpus

## Key Features
- Predictive Model: Hybrid CNN-LSTM architecture for accurate outcome prediction
- Data Visualization: Interactive charts showing:
  - Feature distributions
  - Outcome proportions
  - Gender-based comparisons
  - Age group analysis
- Responsive UI: Clean, mobile-friendly interface
- Data Validation: Input range checking and error handling

## Technical Specifications
### Model Architecture
```python
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

# studentMOOCPredictionUsingDeepLearning
# studentMOOCPredictionUsingDeepLearning

# Student Outcome Prediction System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

![Study Background](https://cdn.canva.com/study-academic-background.jpg) <!-- Replace with actual hosted Canva image if available -->

---

##  Project Title
**Design and Development of a Hybrid Deep Learning Model for Student Performance Prediction in MOOCs: Integrating Optimized Feature Selection**

---

## Project Overview

This system predicts student academic outcomes (**Pass / Fail / Withdrawn / Distinction**) using a **hybrid CNN-LSTM deep learning model**. The system offers:

- 📊 Interactive outcome predictions  
- 🧠 Comprehensive model visualizations  
- 👥 Gender-based performance analysis  
- 📚 Course engagement metrics  
- ✅ Real-time results with validation  

---

## 👨‍💻 Project Contributors

- [**Dr. Shahzad Rizwan**](https://scholar.google.com/citations?user=pewLS_oAAAAJ&hl=en)  
- **Hammad Nawaz**

### 👨‍🏫 Project Advisor
- [**Ts. Dr. Chee Ken Nee**](https://directory.upsi.edu.my/experts/profile/02E6AC7CD9D14955)

---

## 📬 Contact

- 📧 `p20232002567@siswa.upsi.edu.my`  
- 📧 `hammadkhan3923@gmail.com`  
- 📧 `cheekennee@meta.upsi.edu.my`

---

## 🔑 Key Features

- ✅ **Hybrid CNN-LSTM Model**: Robust prediction of outcomes based on optimized features.
- 📈 **Data Visualization**:
  - Feature distribution across inputs
  - Outcome proportions (with minimized pie chart size)
  - Gender-based comparisons
  - Age group-based performance
- 💡 **Model Insights**: Accuracy, Precision, Recall, F1 Score, False Negative Rates
- 🌐 **Responsive UI**: User-friendly and mobile-compatible design
- 🛡️ **Validation**: Input sanitization, range checking, and error handling

---

## 🧠 Model Architecture

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

 
document.addEventListener('DOMContentLoaded', () => {
    // Training data from model output (epochs 1–31, validation accuracy)
    const trainingData = {
        epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        loss: [1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.77, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.60, 0.60, 0.60],
        accuracy: [0.4815, 0.5240, 0.5572, 0.5738, 0.6076, 0.6174, 0.6250, 0.6220, 0.6397, 0.6435, 0.6427, 0.6465, 0.6452, 0.6471, 0.6498, 0.6481, 0.6424, 0.6471, 0.6495, 0.6544, 0.6511, 0.6528, 0.6528, 0.6582, 0.6582, 0.6634, 0.6582, 0.6571, 0.6618, 0.6593, 0.6612],
        feature_importance: {
            'studied_credits': 0.18,
            'forumng': 0.16,
            'oucontent': 0.15,
            'resource': 0.10,
            'highest_education_A Level or Equivalent': 0.08,
            'highest_education_HE Qualification': 0.07,
            'imd_band_0-10%': 0.05,
            'imd_band_90-100%': 0.05,
            'age_band_0-35': 0.06,
            'disability_Y': 0.04,
            'homepage': 0.12,
            'subpage': 0.11,
            'gender_M': 0.06,
            'code_module_AAA': 0.05,
            'date': 0.14,
            'highest_education_Lower Than A Level': 0.07
        }
    };

    // Chart 1: Training Loss (Line)
    {
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: trainingData.epochs,
                datasets: [{
                    label: 'Training Loss',
                    data: trainingData.loss,
                    borderColor: '#a30000',
                    backgroundColor: 'rgba(163, 0, 0, 0.2)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Epoch' } },
                    y: { beginAtZero: true, title: { display: true, text: 'Loss' } }
                },
                plugins: { legend: { position: 'top' } }
            }
        });
    }

    // Chart 2: Training Accuracy (Line)
    {
        const accuracyChart = new Chart(document.getElementById('accuracyChart'), {
            type: 'line',
            data: {
                labels: trainingData.epochs,
                datasets: [{
                    label: 'Training Accuracy',
                    data: trainingData.accuracy,
                    borderColor: '#00a8cc',
                    backgroundColor: 'rgba(0, 168, 204, 0.2)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Epoch' } },
                    y: { beginAtZero: true, max: 1, title: { display: true, text: 'Accuracy' } }
                },
                plugins: { legend: { position: 'top' } }
            }
        });
    }

    // Chart 3: Feature Importance (Scatter)
    {
        const featureImportanceChart = new Chart(document.getElementById('featureImportanceChart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Feature Importance',
                    data: Object.keys(trainingData.feature_importance).map((key, i) => ({
                        x: i,
                        y: trainingData.feature_importance[key]
                    })),
                    backgroundColor: ['#2a9d8f', '#f4a261', '#a30000', '#00a8cc', '#d00000', '#0077b6', 
                                     '#e85d04', '#1b4332', '#ff6384', '#36a2eb', '#4a5568', '#9f7aea',
                                     '#f472b6', '#68d391', '#ed64a6', '#b45309'],
                    borderColor: '#e8f0f2',
                    pointRadius: 8,
                    pointHoverRadius: 12
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        ticks: {
                            callback: function(value, index) {
                                return Object.keys(trainingData.feature_importance)[index] || '';
                            },
                            maxTicksLimit: Object.keys(trainingData.feature_importance).length,
                            autoSkip: false,
                            maxRotation: 45,
                            minRotation: 45
                        },
                        title: { display: true, text: 'Feature' }
                    },
                    y: { beginAtZero: true, title: { display: true, text: 'Importance Score' } }
                },
                plugins: { legend: { position: 'top' } }
            }
        });
    }
});
 
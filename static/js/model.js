document.addEventListener('DOMContentLoaded', () => {
    // Placeholder data (replace with actual training metrics if available)
    const trainingData = {
        epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        loss: [1.2, 0.9, 0.7, 0.55, 0.45, 0.38, 0.32, 0.28, 0.25, 0.22],
        accuracy: [0.45, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90],
        feature_importance: {
            'num_of_prev_attempts': 0.15,
            'studied_credits': 0.20,
            'forumng': 0.18,
            'oucontent': 0.22,
            'quiz': 0.17,
            'resource': 0.14,
            'highest_education_a_level': 0.08,
            'highest_education_he': 0.09,
            'imd_band_0_10': 0.07,
            'imd_band_90_100': 0.06,
            'age_band_0_35': 0.05,
            'disability_y': 0.04
        }
    };

    // Chart 1: Training Loss (Line)
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

    // Chart 2: Training Accuracy (Line)
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

    // Chart 3: Feature Importance (Scatter)
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
                                 '#e85d04', '#1b4332', '#ff6384', '#36a2eb', '#4a5568', '#9f7aea'],
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
});
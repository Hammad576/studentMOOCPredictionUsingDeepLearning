document.addEventListener('DOMContentLoaded', async () => {
    if (document.getElementById('featureChart')) {
        try {
            const response = await fetch('/data');
            const data = await response.json();

            // Chart 1: Bar Chart (Existing)
            const featureChart = new Chart(document.getElementById('featureChart'), {
                type: 'bar',
                data: {
                    labels: ['num_of_prev_attempts', 'studied_credits', 'forumng', 'oucontent', 'quiz', 'resource'],
                    datasets: [{
                        label: 'Average Value',
                        data: [
                            data.feature_means.num_of_prev_attempts,
                            data.feature_means.studied_credits,
                            data.feature_means.forumng,
                            data.feature_means.oucontent,
                            data.feature_means.quiz,
                            data.feature_means.resource
                        ],
                        backgroundColor: '#00a8cc',
                        borderColor: '#007a99',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Average Value' } }
                    },
                    plugins: { legend: { display: true } }
                }
            });

            // Chart 2: Pie Chart (Existing)
            const outcomeChart = new Chart(document.getElementById('outcomeChart'), {
                type: 'pie',
                data: {
                    labels: Object.keys(data.outcome_counts),
                    datasets: [{
                        label: 'Outcome Distribution',
                        data: Object.values(data.outcome_counts),
                        backgroundColor: ['#a30000', '#00a8cc', '#f4a261', '#2a9d8f'],
                        borderColor: '#e8f0f2',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 3: Doughnut Chart (Corrected)
            const outcomeChart2 = new Chart(document.getElementById('outcomeChart2'), {
                type: 'doughnut',
                data: {
                    labels: Object.keys(data.outcome_counts),
                    datasets: [{
                        label: 'Outcome Distribution',
                        data: Object.values(data.outcome_counts),
                        backgroundColor: ['#a30000', '#00a8cc', '#f4a261', '#2a9d8f'],
                        borderColor: '#e8f0f2',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 4: Line Chart (New)
            const lineChart = new Chart(document.getElementById('lineChart'), {
                type: 'line',
                data: {
                    labels: ['num_of_prev_attempts', 'studied_credits', 'forumng', 'oucontent', 'quiz', 'resource'],
                    datasets: [{
                        label: 'Feature Trends',
                        data: [
                            data.feature_means.num_of_prev_attempts,
                            data.feature_means.studied_credits,
                            data.feature_means.forumng,
                            data.feature_means.oucontent,
                            data.feature_means.quiz,
                            data.feature_means.resource
                        ],
                        borderColor: '#2a9d8f',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });

            // Chart 5: Polar Area Chart (New)
            const polarChart = new Chart(document.getElementById('polarChart'), {
                type: 'polarArea',
                data: {
                    labels: Object.keys(data.outcome_counts),
                    datasets: [{
                        label: 'Outcome Distribution',
                        data: Object.values(data.outcome_counts),
                        backgroundColor: ['#a30000', '#00a8cc', '#f4a261', '#2a9d8f']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 6: Radar Chart (New)
            const radarChart = new Chart(document.getElementById('radarChart'), {
                type: 'radar',
                data: {
                    labels: ['num_of_prev_attempts', 'studied_credits', 'forumng', 'oucontent', 'quiz', 'resource'],
                    datasets: [{
                        label: 'Feature Analysis',
                        data: [
                            data.feature_means.num_of_prev_attempts,
                            data.feature_means.studied_credits,
                            data.feature_means.forumng,
                            data.feature_means.oucontent,
                            data.feature_means.quiz,
                            data.feature_means.resource
                        ],
                        backgroundColor: 'rgba(0, 168, 204, 0.2)',
                        borderColor: '#00a8cc'
                    }]
                },
                options: {
                    responsive: true,
                    scales: { r: { beginAtZero: true } }
                }
            });

            // NEW Chart 7: Gender Comparison (Component Bar Chart)
            // In your existing chart.js code, add the gender chart:
            const genderChart = new Chart(document.getElementById('genderChart'), {
                type: 'bar',
                data: {
                    labels: ['Pass', 'Fail', 'Distinction', 'Withdrawn'],
                    datasets: [{
                        label: 'Female',
                        data: [
                            data.gender_counts.female_pass,
                            data.gender_counts.female_fail,
                            data.gender_counts.female_distinction,
                            data.gender_counts.female_withdrawn
                        ],
                        backgroundColor: '#ff6384'
                    }, {
                        label: 'Male',
                        data: [
                            data.gender_counts.male_pass,
                            data.gender_counts.male_fail,
                            data.gender_counts.male_distinction,
                            data.gender_counts.male_withdrawn
                        ],
                        backgroundColor: '#36a2eb'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Number of Students' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });


        } catch (err) {
            console.error('Error loading charts:', err);
            document.getElementById('featureChart').parentElement.innerHTML = '<p>Error loading charts. Please check the dataset.</p>';
        }
    }
});
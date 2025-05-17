document.addEventListener('DOMContentLoaded', async () => {
    if (document.getElementById('featureChart')) {
        try {
            const response = await fetch('/data');
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            const data = await response.json();

            // Chart 1: Feature Distribution (Bar, Unchanged)
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
                        y: { beginAtZero: true, title: { display: true, text: 'Average Value' } },
                        x: { title: { display: true, text: 'Feature' } }
                    },
                    plugins: { legend: { display: true } }
                }
            });

            // Chart 2: Outcome Distribution (Pie → Bar)
            const outcomeChart = new Chart(document.getElementById('outcomeChart'), {
                type: 'bar',
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
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                        x: { title: { display: true, text: 'Outcome' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 3: Outcome Distribution Alternate (Doughnut → Bar)
            const outcomeChart2 = new Chart(document.getElementById('outcomeChart2'), {
                type: 'bar',
                data: {
                    labels: Object.keys(data.outcome_counts),
                    datasets: [{
                        label: 'Outcome Distribution',
                        data: Object.values(data.outcome_counts),
                        backgroundColor: ['#d00000', '#0077b6', '#e85d04', '#1b4332'],
                        borderColor: '#e8f0f2',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                        x: { title: { display: true, text: 'Outcome' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 4: Feature Trends (Line, Unchanged)
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
                        y: { beginAtZero: true, title: { display: true, text: 'Average Value' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 5: Outcome Distribution (PolarArea → Bar)
            const polarChart = new Chart(document.getElementById('polarChart'), {
                type: 'bar',
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
                    scales: {
                        y: { beginAtZero: true, title: { display: true, text: 'Count' } },
                        x: { title: { display: true, text: 'Outcome' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });

            // Chart 6: Gender Comparison (Bar, Unchanged)
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
                        y: { beginAtZero: true, title: { display: true, text: 'Number of Students' } },
                        x: { title: { display: true, text: 'Outcome' } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });

        } catch (err) {
            console.error('Error loading charts:', err);
            const containers = [
                'featureChart', 'outcomeChart', 'outcomeChart2',
                'lineChart', 'polarChart', 'genderChart'
            ];
            containers.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.parentElement.innerHTML = '<p class="text-red-600 text-center">Error loading chart. Please check the dataset.</p>';
            });
        }
    }
});
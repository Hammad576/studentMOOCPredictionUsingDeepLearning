document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            resultDiv.textContent = '';
            errorDiv.textContent = '';

            // Client-side validation
            const inputs = {};
            let isValid = true;
            const ranges = {
                'num_of_prev_attempts': [0, 5],
                'studied_credits': [30, 200],
                'forumng': [0, 100],
                'oucontent': [0, 200],
                'quiz': [0, 50],
                'resource': [0, 100],
                'highest_education_A Level or Equivalent': [0, 1],
                'highest_education_HE Qualification': [0, 1],
                'imd_band_0_10': [0, 1],
                'imd_band_90_100': [0, 1],
                'age_band_0_35': [0, 1],
                'disability_y': [0, 1]
            };

            for (const input of form.elements) {
                if (input.name && input.required) {
                    const value = input.value.trim();
                    if (value === '') {
                        errorDiv.textContent = `Please fill in ${input.name.replace(/_/g, ' ')}`;
                        isValid = false;
                        break;
                    }
                    const numValue = parseFloat(value);
                    if (isNaN(numValue)) {
                        errorDiv.textContent = `${input.name.replace(/_/g, ' ')} must be a number`;
                        isValid = false;
                        break;
                    }
                    const [min, max] = ranges[input.name];
                    if (numValue < min || numValue > max) {
                        errorDiv.textContent = `${input.name.replace(/_/g, ' ')} must be between ${min} and ${max}`;
                        isValid = false;
                        break;
                    }
                    inputs[input.name] = numValue;
                }
            }

            if (!isValid) return;

            // Submit form via AJAX
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams(inputs)
                });
                const data = await response.json();

                // After successful prediction
                if (response.ok) {
                    // Pass data through URL parameters
                    const params = new URLSearchParams({
                        prediction: data.prediction,
                        confidence: data.confidence || 0.85
                    });

                    window.open(`/result?${params}`, '_blank');
                    resultDiv.textContent = 'Prediction submitted successfully. Check the new tab.';
                    resultDiv.classList.remove('hidden');
                } else {
                    errorDiv.textContent = data.error;
                    errorDiv.classList.remove('hidden');
                }
            } catch (err) {
                errorDiv.textContent = 'An error occurred. Please try again.';
                errorDiv.classList.remove('hidden');
            }
        });
    }
});
document.addEventListener("DOMContentLoaded", function() {
    const reviewForm = document.getElementById('review-form');
    const reviewInput = document.getElementById('review');
    const classifyBtn = document.getElementById('classify-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultDiv = document.getElementById('result');
    const sentimentIcon = document.getElementById('sentiment-icon');
    const sentimentText = document.getElementById('sentiment-text');
    const charCounter = document.getElementById('char-counter');

    function updateCharCounter() {
        const currentLength = reviewInput.value.length;
        charCounter.textContent = `${currentLength}`;
        
        if (currentLength > maxChars) {
            charCounter.style.color = 'var(--negative-color)';
        } else {
            charCounter.style.color = 'rgba(255, 255, 255, 0.6)';
        }
    }

    reviewInput.addEventListener('input', updateCharCounter);

    clearBtn.addEventListener('click', function() {
        reviewInput.value = '';
        resultDiv.classList.remove('visible');
        updateCharCounter();
    });

    reviewForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const reviewText = reviewInput.value.trim();

        if (!reviewText) {
            alert('Please enter a review before submitting.');
            return;
        }

        classifyBtn.disabled = true;
        classifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        fetch('/classify/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'review': reviewText
            })
        })
        .then(response => response.json())
        .then(data => {
            const classification = data.classification;
            
            sentimentIcon.innerHTML = classification.toLowerCase() === 'positive' 
                ? '<i class="fas fa-smile-beam positive"></i>' 
                : '<i class="fas fa-frown negative"></i>';
            
            sentimentText.textContent = `The sentiment of your review is ${classification.toLowerCase()}.`;
            sentimentText.className = classification.toLowerCase();

            resultDiv.classList.add('visible');
        })
        .catch(error => {
            console.error('Error:', error);
            sentimentIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            sentimentText.textContent = 'An error occurred. Please try again.';
            sentimentText.className = 'error';
            resultDiv.classList.add('visible');
        })
        .finally(() => {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Sentiment';
        });
    });
});
// Global variables
let analysisResults = null;

// DOM Elements
const form = document.getElementById('geneticForm');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const exportBtn = document.getElementById('exportBtn');
const clearBtn = document.getElementById('clearBtn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkServerHealth();
});

function initializeEventListeners() {
    // Form submission
    form.addEventListener('submit', handleFormSubmit);
    
    // Export button
    exportBtn.addEventListener('click', handleExportReport);
    
    // Clear button
    clearBtn.addEventListener('click', handleClearData);
    
    // Input validation
    const genotypeInputs = document.querySelectorAll('input[name$="_genotype"]');
    genotypeInputs.forEach(input => {
        input.addEventListener('input', validateGenotype);
        input.addEventListener('blur', formatGenotype);
    });
}

function checkServerHealth() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                showError(['Model not loaded. Please contact support.']);
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
            showError(['Unable to connect to analysis server.']);
        });
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    // Clear previous results
    hideResults();
    hideErrors();
    
    // Get form data
    const formData = new FormData(form);
    const data = {
        disorder: formData.get('disorder'),
        parent1_sex: formData.get('parent1_sex'),
        parent1_genotype: formData.get('parent1_genotype'),
        parent1_affected: formData.has('parent1_affected'),
        parent2_sex: formData.get('parent2_sex'),
        parent2_genotype: formData.get('parent2_genotype'),
        parent2_affected: formData.has('parent2_affected'),
        generations: parseInt(formData.get('generations'))
    };
    
    // Validate form data
    const validationErrors = validateFormData(data);
    if (validationErrors.length > 0) {
        showError(validationErrors);
        return;
    }
    
    // Show loading
    showLoading();
    
    // Submit to API
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        
        if (result.success) {
            analysisResults = result.results;
            showResults(result.results);
            exportBtn.disabled = false;
        } else {
            showError(result.errors || [result.error || 'Analysis failed']);
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Analysis failed:', error);
        showError(['Network error. Please try again.']);
    });
}

function handleExportReport() {
    if (!analysisResults) {
        showError(['No analysis results available for export.']);
        return;
    }
    
    showLoading();
    
    fetch('/api/export-report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(analysisResults)
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Export failed');
    })
    .then(blob => {
        hideLoading();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'GrenckDevs_Genetic_Analysis_Report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showSuccessMessage('Report downloaded successfully!');
    })
    .catch(error => {
        hideLoading();
        console.error('Export failed:', error);
        showError(['Failed to export report. Please try again.']);
    });
}

function handleClearData() {
    if (confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
        form.reset();
        hideResults();
        hideErrors();
        analysisResults = null;
        exportBtn.disabled = true;
        
        showSuccessMessage('All data has been cleared successfully.');
    }
}

function validateFormData(data) {
    const errors = [];
    
    if (!data.disorder || data.disorder === '') {
        errors.push('Please select a genetic disorder');
    }
    
    if (!data.parent1_sex) {
        errors.push('Parent 1: Please select biological sex');
    }
    
    if (!data.parent2_sex) {
        errors.push('Parent 2: Please select biological sex');
    }
    
    if (!data.parent1_genotype || data.parent1_genotype.length !== 2) {
        errors.push('Parent 1: Genotype must be exactly 2 characters (e.g., CC, CT, TT)');
    }
    
    if (!data.parent2_genotype || data.parent2_genotype.length !== 2) {
        errors.push('Parent 2: Genotype must be exactly 2 characters (e.g., CC, CT, TT)');
    }
    
    if (!data.generations || data.generations < 1 || data.generations > 10) {
        errors.push('Generations must be between 1 and 10');
    }
    
    return errors;
}

function validateGenotype(event) {
    const input = event.target;
    const value = input.value.toUpperCase();
    
    // Only allow letters
    const validValue = value.replace(/[^A-Z]/g, '');
    
    if (validValue !== value) {
        input.value = validValue;
    }
    
    // Limit to 2 characters
    if (input.value.length > 2) {
        input.value = input.value.substring(0, 2);
    }
}

function formatGenotype(event) {
    const input = event.target;
    input.value = input.value.toUpperCase();
}

function showResults(results) {
    // Update result values
    document.getElementById('resultDisorder').textContent = results.disorder;
    document.getElementById('resultProbability').textContent = `${results.probability.toFixed(2)}%`;
    
    // Update parameters
    const parametersDiv = document.getElementById('resultParameters');
    parametersDiv.innerHTML = `
        <p>• Parent 1: ${results.input_data.parent1_sex} (${results.input_data.parent1_genotype})</p>
        <p>• Parent 2: ${results.input_data.parent2_sex} (${results.input_data.parent2_genotype})</p>
        <p>• Generations Simulated: ${results.input_data.generations}</p>
    `;
    
    // Update Punnett square image
    const punnettImage = document.getElementById('punnettImage');
    punnettImage.src = `data:image/png;base64,${results.punnett_image}`;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(errors) {
    const errorList = document.getElementById('errorList');
    errorList.innerHTML = '';
    
    errors.forEach(error => {
        const li = document.createElement('li');
        li.textContent = error;
        errorList.appendChild(li);
    });
    
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function hideErrors() {
    errorSection.style.display = 'none';
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showSuccessMessage(message) {
    // Create temporary success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    successDiv.style.position = 'fixed';
    successDiv.style.top = '20px';
    successDiv.style.right = '20px';
    successDiv.style.zIndex = '10000';
    successDiv.style.maxWidth = '400px';
    
    document.body.appendChild(successDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (document.body.contains(successDiv)) {
            document.body.removeChild(successDiv);
        }
    }, 3000);
}

// Utility functions
function formatProbability(probability) {
    if (probability >= 75) return 'High Risk';
    if (probability >= 25) return 'Moderate Risk';
    return 'Low Risk';
}

function getProbabilityColor(probability) {
    if (probability >= 75) return '#8b6b6b';
    if (probability >= 25) return '#8b8b6b';
    return '#6b8b6b';
}
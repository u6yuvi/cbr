// API configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const createTenantBtn = document.getElementById('createTenant');
const tenantLoader = document.getElementById('tenantLoader');
const tenantInfo = document.getElementById('tenantInfo');
const currentTenantId = document.getElementById('currentTenantId');
const classSection = document.getElementById('classSection');
const predictionSection = document.getElementById('predictionSection');
const classList = document.getElementById('classList');
const dogImagesContainer = document.getElementById('dogImages');
const catImagesContainer = document.getElementById('catImages');
const addDogClassBtn = document.getElementById('addDogClass');
const addCatClassBtn = document.getElementById('addCatClass');
const dogLoader = document.getElementById('dogLoader');
const catLoader = document.getElementById('catLoader');
const exampleImagesContainer = document.getElementById('exampleImages');
const selectedExampleName = document.getElementById('selectedExampleName');
const makePredictionBtn = document.getElementById('makePrediction');
const predictionLoader = document.getElementById('predictionLoader');
const predictionResult = document.getElementById('predictionResult');
const predictedClass = document.getElementById('predictedClass');
const confidence = document.getElementById('confidence');
const probabilities = document.getElementById('probabilities');

// State management
let currentTenant = null;
let availableClasses = new Set();
let selectedExampleImage = null;

// Helper functions
function showLoader(loader) {
    loader.classList.remove('hidden');
}

function hideLoader(loader) {
    loader.classList.add('hidden');
}

function showElement(element) {
    element.classList.remove('hidden');
}

function hideElement(element) {
    element.classList.add('hidden');
}

async function makeRequest(endpoint, options = {}) {
    const headers = {
        ...(currentTenant && { 'X-Tenant-ID': currentTenant })
    };

    if (!options.files) {
        headers['Content-Type'] = 'application/json';
    }

    try {
        // Check if we're in simulation mode
        if (options.body && typeof options.body === 'string' && 
            JSON.parse(options.body).simulation) {
            
            // Return simulated responses for demonstration
            console.log(`Simulating API call to ${endpoint}`);
            
            // Simulate a delay
            await new Promise(resolve => setTimeout(resolve, 500));
            
            if (endpoint.includes('/class/add/dog')) {
                return { success: true, message: 'Dog class added successfully' };
            }
            
            if (endpoint.includes('/class/add/cat')) {
                return { success: true, message: 'Cat class added successfully' };
            }
            
            if (endpoint === '/predict') {
                // This should never be called in simulation mode
                // as we handle predictions differently
                return {
                    predicted_class: 'unknown',
                    confidence: 0.5,
                    class_probabilities: { unknown: 0.5 }
                };
            }
            
            return { success: true };
        }
        
        // Make the actual API request
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers: {
                ...headers,
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }

        return response.json();
    } catch (error) {
        console.error(`API request error for ${endpoint}:`, error);
        throw error;
    }
}

// Save state to Chrome storage
function saveState() {
    chrome.storage.local.set({
        tenantId: currentTenant,
        availableClasses: Array.from(availableClasses)
    });
}

// Load state from Chrome storage
async function loadState() {
    return new Promise((resolve) => {
        chrome.storage.local.get(['tenantId', 'availableClasses'], (result) => {
            if (result.tenantId) {
                currentTenant = result.tenantId;
                currentTenantId.textContent = currentTenant;
                showElement(tenantInfo);
                showElement(classSection);
                showElement(predictionSection);
            }

            if (result.availableClasses) {
                availableClasses = new Set(result.availableClasses);
                updateClassList();
            }

            resolve();
        });
    });
}

// Event handlers
createTenantBtn.addEventListener('click', async () => {
    try {
        showLoader(tenantLoader);
        
        // In a real implementation, we would make an actual API call
        // Here we're simulating tenant creation for demonstration
        const response = {
            tenant_id: 'demo-tenant-' + Date.now().toString().slice(-6)
        };

        currentTenant = response.tenant_id;
        currentTenantId.textContent = currentTenant;
        showElement(tenantInfo);
        showElement(classSection);
        showElement(predictionSection);

        saveState();
    } catch (error) {
        console.error('Failed to create tenant:', error);
        alert('Failed to create tenant. Please try again.');
    } finally {
        hideLoader(tenantLoader);
    }
});

// Add dog class with preloaded images
addDogClassBtn.addEventListener('click', async () => {
    if (!currentTenant) {
        alert('Please create a tenant first');
        return;
    }

    try {
        showLoader(dogLoader);
        
        // In a real implementation, we would get the actual image data
        // Here we're simulating the addition of "dog" class with dummy data
        await makeRequest(`/class/add/dog`, {
            method: 'POST',
            body: JSON.stringify({ 
                simulation: true, 
                message: 'Adding dog class' 
            })
        });
        
        // Update available classes
        availableClasses.add('dog');
        updateClassList();
        saveState();
        
        // Show success message
        alert('Successfully added dog class');
    } catch (error) {
        console.error('Failed to add dog class:', error);
        alert('Failed to add dog class. Please try again.');
    } finally {
        hideLoader(dogLoader);
    }
});

// Add cat class with preloaded images
addCatClassBtn.addEventListener('click', async () => {
    if (!currentTenant) {
        alert('Please create a tenant first');
        return;
    }

    try {
        showLoader(catLoader);
        
        // In a real implementation, we would get the actual image data
        // Here we're simulating the addition of "cat" class with dummy data
        await makeRequest(`/class/add/cat`, {
            method: 'POST',
            body: JSON.stringify({ 
                simulation: true, 
                message: 'Adding cat class' 
            })
        });
        
        // Update available classes
        availableClasses.add('cat');
        updateClassList();
        saveState();
        
        // Show success message
        alert('Successfully added cat class');
    } catch (error) {
        console.error('Failed to add cat class:', error);
        alert('Failed to add cat class. Please try again.');
    } finally {
        hideLoader(catLoader);
    }
});

// Handle example image selection
document.querySelectorAll('.example-image').forEach(img => {
    img.addEventListener('click', () => {
        // Remove selection from all images
        document.querySelectorAll('.example-image').forEach(image => {
            image.classList.remove('selected');
        });
        
        // Add selection to clicked image
        img.classList.add('selected');
        selectedExampleImage = img;
        selectedExampleName.textContent = `Selected: ${img.dataset.name}`;
    });
});

// Make prediction with selected example image
makePredictionBtn.addEventListener('click', async () => {
    if (!currentTenant) {
        alert('Please create a tenant first');
        return;
    }
    
    if (!selectedExampleImage) {
        alert('Please select an example image first');
        return;
    }
    
    try {
        showLoader(predictionLoader);
        
        // In a real implementation, we would use the actual image data
        // Here we're simulating a prediction with dummy data based on the selected image
        const imageName = selectedExampleImage.dataset.name.toLowerCase();
        
        // Simulate API response based on selected image
        let simulatedResponse;
        if (imageName.includes('dog')) {
            simulatedResponse = {
                predicted_class: 'dog',
                confidence: 0.85,
                class_probabilities: {
                    dog: 0.85,
                    cat: 0.15
                }
            };
        } else {
            simulatedResponse = {
                predicted_class: 'cat',
                confidence: 0.92,
                class_probabilities: {
                    cat: 0.92,
                    dog: 0.08
                }
            };
        }
        
        // Display results
        predictedClass.textContent = `Predicted Class: ${simulatedResponse.predicted_class}`;
        confidence.textContent = `Confidence: ${(simulatedResponse.confidence * 100).toFixed(2)}%`;
        
        // Display class probabilities
        probabilities.innerHTML = Object.entries(simulatedResponse.class_probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([className, prob]) => `
                <div class="flex justify-between items-center mb-1">
                    <span class="text-sm">${className}</span>
                    <span class="text-sm text-gray-600">${(prob * 100).toFixed(2)}%</span>
                </div>
            `).join('');
        
        showElement(predictionResult);
    } catch (error) {
        console.error('Failed to make prediction:', error);
        alert('Failed to make prediction. Please try again.');
    } finally {
        hideLoader(predictionLoader);
    }
});

function updateClassList() {
    classList.innerHTML = Array.from(availableClasses)
        .map(className => `
            <div class="flex items-center justify-between p-2 bg-white rounded shadow">
                <span class="font-medium">${className}</span>
                <button class="text-red-500 hover:text-red-700" onclick="removeClass('${className}')">
                    Remove
                </button>
            </div>
        `).join('');
}

// Define global removeClass function
window.removeClass = async function(className) {
    try {
        // In a real implementation, we would make an actual API call
        // Here we're simulating class removal for demonstration
        console.log(`Simulating removal of class: ${className}`);
        
        // Simulate a delay
        await new Promise(resolve => setTimeout(resolve, 300));
        
        availableClasses.delete(className);
        updateClassList();
        saveState();
        
        alert(`Class '${className}' removed successfully`);
    } catch (error) {
        console.error('Failed to remove class:', error);
        alert('Failed to remove class. Please try again.');
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadState();
        
        // Load available classes if we have a tenant
        if (currentTenant) {
            try {
                // In a real implementation, we would make an actual API call
                // Here we're simulating model info for demonstration
                const modelInfo = {
                    available_classes: Array.from(availableClasses)
                };
                
                if (modelInfo && modelInfo.available_classes) {
                    availableClasses = new Set(modelInfo.available_classes);
                    updateClassList();
                }
            } catch (error) {
                console.error('Failed to load model info:', error);
            }
        }
    } catch (error) {
        console.error('Failed to initialize:', error);
    }
}); 
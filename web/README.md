# CbR Web Interface

This is a web interface for the Classification by Retrieval (CbR) model, allowing you to run the model directly in your browser using ONNX Runtime Web.

## Features

- Run the CbR model entirely in the browser
- Upload images or use webcam for classification
- Real-time predictions
- Modern, responsive UI using Material-UI
- No server required after initial model loading

## Setup

1. **Export the Model**
   First, export your trained CbR model to ONNX format:
   ```bash
   python export_onnx.py
   ```
   This will create the following files in `web/public/model/`:
   - `backbone.onnx`: The ONNX model file
   - `index_data.json`: Index embeddings and class mappings
   - `preprocess_params.json`: Image preprocessing parameters

2. **Install Dependencies**
   ```bash
   cd web
   npm install
   ```

3. **Start Development Server**
   ```bash
   npm start
   ```
   The application will be available at `http://localhost:3000`

4. **Build for Production**
   ```bash
   npm run build
   ```
   The build files will be in the `build` directory.

## Usage

1. **Upload an Image**
   - Click the "Upload Image" button to select an image file
   - The model will process the image and display predictions

2. **Use Webcam**
   - Allow camera access when prompted
   - Click the "Capture" button to take a photo
   - The model will process the captured image and display predictions

## Technical Details

### Model Architecture

The web interface implements the same classification pipeline as the Python version:

1. **Image Preprocessing**
   - Resize to 224x224
   - Normalize using ImageNet mean and std
   - Convert to correct tensor format

2. **Feature Extraction**
   - Run the backbone model using ONNX Runtime Web
   - Get embedding vector

3. **Similarity Computation**
   - Compute similarities with index embeddings
   - Aggregate results per class
   - Apply softmax to get probabilities

### Performance Considerations

- The model runs entirely in the browser using WebGL acceleration when available
- Initial model loading might take a few seconds
- Subsequent predictions are fast
- Model size is optimized for web deployment

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

1. **Model Loading Issues**
   - Check that model files are in the correct location
   - Ensure CORS is properly configured if serving from a different domain
   - Check browser console for detailed error messages

2. **Webcam Issues**
   - Ensure camera permissions are granted
   - Try a different browser if issues persist
   - Check if your device has a working camera

3. **Performance Issues**
   - Try reducing the webcam resolution
   - Ensure WebGL is enabled in your browser
   - Close other resource-intensive tabs 
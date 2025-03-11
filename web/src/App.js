import React, { useState, useEffect, useRef } from 'react';
import { 
  Container, Paper, Typography, Button, Box, CircularProgress,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  List, ListItem, ListItemText, ListItemSecondaryAction,
  IconButton, Tabs, Tab, Chip, Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';
import Webcam from 'react-webcam';
import * as ort from 'onnxruntime-web';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';

// Styled components
const Input = styled('input')({
  display: 'none',
});

const ImagePreview = styled('img')({
  maxWidth: '100%',
  maxHeight: '300px',
  objectFit: 'contain',
});

const ClassImage = styled('img')({
  width: '100px',
  height: '100px',
  objectFit: 'cover',
  margin: '4px',
  borderRadius: '4px',
});

function App() {
  const [session, setSession] = useState(null);
  const [indexData, setIndexData] = useState(null);
  const [preprocessParams, setPreprocessParams] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const webcamRef = useRef(null);
  
  // New state for index management
  const [activeTab, setActiveTab] = useState(0);
  const [addClassDialogOpen, setAddClassDialogOpen] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedClass, setSelectedClass] = useState(null);
  const [classImages, setClassImages] = useState({});

  // Load ONNX model and metadata
  useEffect(() => {
    async function loadModel() {
      try {
        // Load ONNX model
        const model = await ort.InferenceSession.create('/model/backbone.onnx');
        setSession(model);

        // Load index data
        const indexResponse = await fetch('/model/index_data.json');
        const indexData = await indexResponse.json();
        setIndexData(indexData);

        // Load preprocessing parameters
        const paramsResponse = await fetch('/model/preprocess_params.json');
        const params = await paramsResponse.json();
        setPreprocessParams(params);
      } catch (error) {
        console.error('Error loading model:', error);
      }
    }
    loadModel();
  }, []);

  // Save index data
  const saveIndexData = async () => {
    try {
      const response = await fetch('/model/index_data.json', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(indexData),
      });
      if (!response.ok) throw new Error('Failed to save index data');
    } catch (error) {
      console.error('Error saving index data:', error);
    }
  };

  // Add new class
  const handleAddClass = async () => {
    if (!newClassName || selectedFiles.length === 0) return;

    setLoading(true);
    try {
      const embeddings = [];
      const imageUrls = [];

      // Process each image
      for (const file of selectedFiles) {
        const imageUrl = URL.createObjectURL(file);
        imageUrls.push(imageUrl);

        const img = new Image();
        img.src = imageUrl;
        await new Promise(resolve => img.onload = resolve);

        const preprocessedData = await preprocessImage(img);
        const embedding = await getEmbedding(preprocessedData);
        embeddings.push(Array.from(embedding));
      }

      // Update index data
      const newClassIdx = indexData.num_classes;
      const newLabels = Array(selectedFiles.length).fill(newClassIdx);

      const updatedIndexData = {
        ...indexData,
        embeddings: [...indexData.embeddings, ...embeddings],
        labels: [...indexData.labels, ...newLabels],
        classes_to_idx: {
          ...indexData.classes_to_idx,
          [newClassName]: newClassIdx,
        },
        idx_to_classes: {
          ...indexData.idx_to_classes,
          [newClassIdx]: newClassName,
        },
        num_classes: indexData.num_classes + 1,
      };

      setIndexData(updatedIndexData);
      setClassImages(prev => ({
        ...prev,
        [newClassName]: imageUrls,
      }));

      await saveIndexData();
      setAddClassDialogOpen(false);
      setNewClassName('');
      setSelectedFiles([]);
    } catch (error) {
      console.error('Error adding class:', error);
    } finally {
      setLoading(false);
    }
  };

  // Delete class
  const handleDeleteClass = async (className) => {
    setLoading(true);
    try {
      const classIdx = indexData.classes_to_idx[className];
      
      // Filter out embeddings and labels for this class
      const newEmbeddings = [];
      const newLabels = [];
      indexData.embeddings.forEach((embedding, i) => {
        if (indexData.labels[i] !== classIdx) {
          newEmbeddings.push(embedding);
          newLabels.push(indexData.labels[i]);
        }
      });

      // Update class mappings
      const { [className]: removed, ...newClassesToIdx } = indexData.classes_to_idx;
      const { [classIdx]: removedIdx, ...newIdxToClasses } = indexData.idx_to_classes;

      // Remap indices
      const updatedIndexData = {
        ...indexData,
        embeddings: newEmbeddings,
        labels: newLabels,
        classes_to_idx: newClassesToIdx,
        idx_to_classes: newIdxToClasses,
        num_classes: indexData.num_classes - 1,
      };

      setIndexData(updatedIndexData);
      const { [className]: removedImages, ...newClassImages } = classImages;
      setClassImages(newClassImages);

      await saveIndexData();
    } catch (error) {
      console.error('Error deleting class:', error);
    } finally {
      setLoading(false);
    }
  };

  // Delete sample
  const handleDeleteSample = async (className, imageIndex) => {
    setLoading(true);
    try {
      const classIdx = indexData.classes_to_idx[className];
      
      // Find the index in the global embeddings array
      const globalIndex = indexData.labels.findIndex((label, idx) => 
        label === classIdx && !indexData.embeddings.slice(0, idx)
          .filter((_,i) => indexData.labels[i] === classIdx)
          .length === imageIndex
      );

      if (globalIndex === -1) return;

      // Update embeddings and labels
      const newEmbeddings = [
        ...indexData.embeddings.slice(0, globalIndex),
        ...indexData.embeddings.slice(globalIndex + 1)
      ];
      const newLabels = [
        ...indexData.labels.slice(0, globalIndex),
        ...indexData.labels.slice(globalIndex + 1)
      ];

      const updatedIndexData = {
        ...indexData,
        embeddings: newEmbeddings,
        labels: newLabels,
      };

      setIndexData(updatedIndexData);
      
      // Update class images
      const newClassImages = [...classImages[className]];
      newClassImages.splice(imageIndex, 1);
      setClassImages(prev => ({
        ...prev,
        [className]: newClassImages,
      }));

      await saveIndexData();
    } catch (error) {
      console.error('Error deleting sample:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get embedding for a single image
  const getEmbedding = async (preprocessedData) => {
    const inputTensor = new ort.Tensor(
      'float32',
      preprocessedData,
      [1, 3, preprocessParams.size, preprocessParams.size]
    );

    const output = await session.run({ input: inputTensor });
    return output.embedding.data;
  };

  // Preprocess image
  const preprocessImage = async (imageData) => {
    const { mean, std, size } = preprocessParams;
    
    // Create a canvas to resize the image
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    // Draw and resize image
    ctx.drawImage(imageData, 0, 0, size, size);
    
    // Get image data
    const imageDataArray = ctx.getImageData(0, 0, size, size).data;
    
    // Convert to float32 and normalize
    const float32Data = new Float32Array(3 * size * size);
    for (let i = 0; i < size * size; i++) {
      for (let c = 0; c < 3; c++) {
        float32Data[c * size * size + i] = 
          (imageDataArray[i * 4 + c] / 255.0 - mean[c]) / std[c];
      }
    }
    
    return float32Data;
  };

  // Run inference
  const runInference = async (preprocessedData) => {
    try {
      // Create input tensor
      const inputTensor = new ort.Tensor(
        'float32',
        preprocessedData,
        [1, 3, preprocessParams.size, preprocessParams.size]
      );

      // Run inference
      const output = await session.run({ input: inputTensor });
      const embedding = output.embedding.data;

      // Compute similarities with index embeddings
      const similarities = computeSimilarities(embedding, indexData.embeddings);
      
      // Aggregate results per class
      const predictions = aggregateResults(similarities, indexData);

      return predictions;
    } catch (error) {
      console.error('Inference error:', error);
      throw error;
    }
  };

  // Compute similarities between input embedding and index embeddings
  const computeSimilarities = (embedding, indexEmbeddings) => {
    const similarities = [];
    for (const indexEmbedding of indexEmbeddings) {
      let similarity = 0;
      for (let i = 0; i < embedding.length; i++) {
        similarity += embedding[i] * indexEmbedding[i];
      }
      similarities.push(similarity);
    }
    return similarities;
  };

  // Aggregate results per class
  const aggregateResults = (similarities, indexData) => {
    const classScores = {};
    
    // Initialize scores
    for (const className of Object.keys(indexData.classes_to_idx)) {
      classScores[className] = -Infinity;
    }
    
    // Get max similarity per class
    similarities.forEach((similarity, i) => {
      const classIdx = indexData.labels[i];
      const className = indexData.idx_to_classes[classIdx];
      classScores[className] = Math.max(classScores[className], similarity);
    });
    
    // Convert to probabilities using softmax
    const scores = Object.values(classScores);
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExpScores = expScores.reduce((a, b) => a + b, 0);
    
    const predictions = {};
    Object.keys(classScores).forEach((className, i) => {
      predictions[className] = expScores[i] / sumExpScores;
    });
    
    return predictions;
  };

  // Handle file upload
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setLoading(true);
      try {
        // Load and display image
        const imageUrl = URL.createObjectURL(file);
        setSelectedImage(imageUrl);
        
        // Load image data
        const img = new Image();
        img.src = imageUrl;
        await new Promise(resolve => img.onload = resolve);
        
        // Preprocess image
        const preprocessedData = await preprocessImage(img);
        
        // Run inference
        const predictions = await runInference(preprocessedData);
        setPrediction(predictions);
      } catch (error) {
        console.error('Error processing image:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  // Capture from webcam
  const captureImage = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setLoading(true);
      try {
        setSelectedImage(imageSrc);
        
        // Load image data
        const img = new Image();
        img.src = imageSrc;
        await new Promise(resolve => img.onload = resolve);
        
        // Preprocess image
        const preprocessedData = await preprocessImage(img);
        
        // Run inference
        const predictions = await runInference(preprocessedData);
        setPrediction(predictions);
      } catch (error) {
        console.error('Error processing webcam image:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Image Classification
        </Typography>

        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} sx={{ mb: 2 }}>
          <Tab label="Classify" />
          <Tab label="Manage Classes" />
        </Tabs>

        {activeTab === 0 ? (
          // Classification tab
          <>
            {/* Keep existing classification UI */}
            <Box sx={{ my: 2 }}>
              <label htmlFor="image-upload">
                <Input
                  accept="image/*"
                  id="image-upload"
                  type="file"
                  onChange={handleImageUpload}
                  disabled={!session}
                />
                <Button
                  variant="contained"
                  component="span"
                  disabled={!session}
                >
                  Upload Image
                </Button>
              </label>
            </Box>

            <Box sx={{ my: 2 }}>
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                style={{ width: '100%' }}
              />
              <Button
                variant="contained"
                onClick={captureImage}
                disabled={!session}
                sx={{ mt: 1 }}
              >
                Capture
              </Button>
            </Box>

            {selectedImage && (
              <Paper sx={{ p: 2, my: 2 }}>
                <ImagePreview src={selectedImage} alt="Preview" />
              </Paper>
            )}

            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                <CircularProgress />
              </Box>
            )}

            {prediction && (
              <Paper sx={{ p: 2, my: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Predictions
                </Typography>
                {Object.entries(prediction)
                  .sort(([, a], [, b]) => b - a)
                  .map(([className, probability]) => (
                    <Typography key={className}>
                      {className}: {(probability * 100).toFixed(2)}%
                    </Typography>
                  ))}
              </Paper>
            )}
          </>
        ) : (
          // Class management tab
          <Box>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setAddClassDialogOpen(true)}
              sx={{ mb: 2 }}
            >
              Add New Class
            </Button>

            <Grid container spacing={2}>
              {indexData && Object.keys(indexData.classes_to_idx).map(className => (
                <Grid item xs={12} md={6} key={className}>
                  <Paper sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6">{className}</Typography>
                      <IconButton
                        onClick={() => handleDeleteClass(className)}
                        color="error"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                    
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {classImages[className]?.map((imageUrl, idx) => (
                        <Box key={idx} sx={{ position: 'relative' }}>
                          <ClassImage src={imageUrl} alt={`${className} ${idx + 1}`} />
                          <IconButton
                            size="small"
                            sx={{
                              position: 'absolute',
                              top: 0,
                              right: 0,
                              backgroundColor: 'rgba(255, 255, 255, 0.8)',
                            }}
                            onClick={() => handleDeleteSample(className, idx)}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>

            {/* Add Class Dialog */}
            <Dialog open={addClassDialogOpen} onClose={() => setAddClassDialogOpen(false)}>
              <DialogTitle>Add New Class</DialogTitle>
              <DialogContent>
                <TextField
                  autoFocus
                  margin="dense"
                  label="Class Name"
                  fullWidth
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                />
                <Box sx={{ mt: 2 }}>
                  <label htmlFor="class-images">
                    <Input
                      accept="image/*"
                      id="class-images"
                      multiple
                      type="file"
                      onChange={(e) => setSelectedFiles(Array.from(e.target.files))}
                    />
                    <Button variant="outlined" component="span">
                      Select Images
                    </Button>
                  </label>
                  {selectedFiles.length > 0 && (
                    <Typography sx={{ mt: 1 }}>
                      {selectedFiles.length} images selected
                    </Typography>
                  )}
                </Box>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setAddClassDialogOpen(false)}>Cancel</Button>
                <Button onClick={handleAddClass} disabled={!newClassName || selectedFiles.length === 0}>
                  Add Class
                </Button>
              </DialogActions>
            </Dialog>
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default App; 
import * as tf from '@tensorflow/tfjs';

// Model URL for a MobileNetV2 model
const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json';

// Mapping of model output indices to dog breeds
const BREED_CLASSES = [
  'Labrador',
  'Golden Retriever',
  'German Shepherd',
  'Bulldog',
  'Poodle',
  'Beagle',
  'Rottweiler',
  'Yorkshire Terrier',
  'Boxer',
  'Dachshund'
];

let modelPromise: Promise<tf.GraphModel> | null = null;

async function loadModel(): Promise<tf.GraphModel> {
  if (!modelPromise) {
    try {
      console.log('Loading model...');
      modelPromise = tf.loadGraphModel(MODEL_URL);
      await modelPromise; // Ensure model loads
      console.log('Model loaded successfully');
      return modelPromise;
    } catch (error) {
      console.error('Model loading error:', error);
      modelPromise = null;
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }
  return modelPromise;
}

async function preprocessImage(file: File): Promise<tf.Tensor> {
  try {
    console.log('Starting image preprocessing...');
    const img = await createImageBitmap(file);
    return tf.tidy(() => {
      console.log('Converting image to tensor...');
      const imgTensor = tf.browser.fromPixels(img)
        .resizeBilinear([224, 224])
        .toFloat()
        .sub(127.5)
        .div(127.5)
        .expandDims();
      console.log('Image preprocessing complete');
      return imgTensor;
    });
  } catch (error) {
    console.error('Image preprocessing error:', error);
    throw new Error(`Failed to preprocess image: ${error.message}`);
  }
}

export async function classifyImage(file: File) {
  let imageTensor: tf.Tensor | null = null;
  let predictions: tf.Tensor | null = null;
  let probabilities: tf.Tensor | null = null;

  try {
    console.log('Starting classification process...');
    tf.engine().startScope();
    
    const model = await loadModel();
    console.log('Model loaded, preprocessing image...');
    imageTensor = await preprocessImage(file);
    
    // Get predictions
    console.log('Running inference...');
    predictions = model.predict(imageTensor) as tf.Tensor;
    probabilities = tf.softmax(predictions);
    
    console.log('Getting prediction probabilities...');
    const values = await probabilities.data();

    // Get top 3 predictions
    console.log('Processing prediction results...');
    const topPredictions = Array.from(values)
      .map((confidence, index) => ({
        breed: BREED_CLASSES[index] || 'Unknown Breed',
        confidence: parseFloat(confidence.toFixed(4))
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    console.log('Top predictions:', topPredictions);

    // Generate reference image URLs based on breed names
    const referenceImages = topPredictions.map(pred => 
      `https://source.unsplash.com/featured/?${encodeURIComponent(pred.breed.toLowerCase())},dog`
    );

    console.log('Classification complete');
    return {
      predictions: topPredictions,
      referenceImages
    };

  } catch (error) {
    console.error('Classification error:', error);
    throw new Error(`Failed to classify image: ${error.message}`);
  } finally {
    console.log('Cleaning up tensors...');
    // Clean up all tensors
    if (imageTensor) tf.dispose(imageTensor);
    if (predictions) tf.dispose(predictions);
    if (probabilities) tf.dispose(probabilities);
    tf.engine().endScope();
    console.log('Cleanup complete');
  }
}

// Memory management utility
export function clearTensorMemory() {
  tf.engine().endScope();
  tf.engine().startScope();
}

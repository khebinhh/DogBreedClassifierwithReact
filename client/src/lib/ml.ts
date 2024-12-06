import * as tf from '@tensorflow/tfjs';

// Model URL for a MobileNetV2 model fine-tuned on dog breeds
const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json';

// Mapping of model output indices to dog breeds (simplified for demo)
const BREED_CLASSES = [
  'Labrador Retriever',
  'German Shepherd',
  'Golden Retriever',
  'Bulldog',
  'Beagle',
  // Add more breeds as needed
];

let modelPromise: Promise<tf.GraphModel> | null = null;

async function loadModel(): Promise<tf.GraphModel> {
  if (!modelPromise) {
    try {
      modelPromise = tf.loadGraphModel(MODEL_URL);
      await modelPromise; // Ensure model loads successfully
    } catch (error) {
      modelPromise = null; // Reset on error
      console.error('Model loading error:', error);
      throw new Error('Failed to load the ML model. Please try again later.');
    }
  }
  return modelPromise;
}

async function preprocessImage(file: File): Promise<tf.Tensor> {
  try {
    const img = await createImageBitmap(file);
    const tensor = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(img)
        .resizeBilinear([224, 224])
        .expandDims()
        .toFloat()
        .div(255.0);
      return imgTensor;
    });
    return tensor;
  } catch (error) {
    throw new Error(`Failed to preprocess image: ${error}`);
  }
}

export async function classifyImage(file: File) {
  let imageTensor: tf.Tensor | null = null;
  let predictions: tf.Tensor | null = null;

  try {
    const model = await loadModel();
    imageTensor = await preprocessImage(file);
    
    // Get predictions
    predictions = model.predict(imageTensor) as tf.Tensor;
    const probabilities = await predictions.data();

    // Get top 3 predictions
    const topPredictions = Array.from(probabilities)
      .map((confidence, index) => ({
        breed: BREED_CLASSES[index] || 'Unknown Breed',
        confidence: confidence
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    // Sample reference images (using provided stock photos)
    const referenceImages = [
      `https://images.unsplash.com/photo-${topPredictions[0].breed.toLowerCase().replace(/\s+/g, '-')}`,
      `https://images.unsplash.com/photo-${topPredictions[1].breed.toLowerCase().replace(/\s+/g, '-')}`,
      `https://images.unsplash.com/photo-${topPredictions[2].breed.toLowerCase().replace(/\s+/g, '-')}`
    ];

    return {
      predictions: topPredictions,
      referenceImages
    };

  } catch (error) {
    console.error('Classification error:', error);
    throw new Error('Failed to classify image');
  } finally {
    // Clean up tensors
    if (imageTensor) imageTensor.dispose();
    if (predictions) predictions.dispose();
    tf.engine().endScope(); // Ensure all intermediate tensors are cleaned up
  }
}

// Memory management utility
export function clearTensorMemory() {
  tf.engine().endScope();
  tf.engine().startScope();
}

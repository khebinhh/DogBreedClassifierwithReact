import * as tf from '@tensorflow/tfjs';

// Model URL for a MobileNetV2 model fine-tuned on dog breeds
const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/mobilenet_v2_130_224/classification/3/default/1';

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
    modelPromise = tf.loadGraphModel(MODEL_URL, {fromTFHub: true})
      .catch(error => {
        modelPromise = null;
        console.error('Model loading error:', error);
        throw new Error(`Failed to load model: ${error.message}`);
      });
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
  let intermediateTensors: tf.Tensor[] = [];

  try {
    // Start a new scope to better manage memory
    tf.engine().startScope();
    
    const model = await loadModel();
    imageTensor = await preprocessImage(file);
    
    // Get predictions
    predictions = tf.tidy(() => {
      const pred = model.predict(imageTensor!) as tf.Tensor;
      // Softmax to get probabilities
      return tf.softmax(pred);
    });
    
    const probabilities = await predictions.data();

    // Get top 3 predictions
    const topPredictions = Array.from(probabilities)
      .map((confidence, index) => ({
        breed: BREED_CLASSES[index] || 'Unknown Breed',
        confidence: parseFloat(confidence.toFixed(4))
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    // Generate reference image URLs based on breed names
    const referenceImages = topPredictions.map(pred => 
      `https://source.unsplash.com/featured/?${encodeURIComponent(pred.breed.toLowerCase())},dog`
    );

    return {
      predictions: topPredictions,
      referenceImages
    };

  } catch (error) {
    console.error('Classification error:', error);
    throw new Error('Failed to classify image. Please try again with a different image.');
  } finally {
    // Clean up all tensors
    if (imageTensor) {
      tf.dispose(imageTensor);
    }
    if (predictions) {
      tf.dispose(predictions);
    }
    intermediateTensors.forEach(tensor => {
      if (tensor) tf.dispose(tensor);
    });
    tf.engine().endScope();
  }
}

// Memory management utility
export function clearTensorMemory() {
  tf.engine().endScope();
  tf.engine().startScope();
}

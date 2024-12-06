// Types for classification results
interface Prediction {
  breed: string;
  confidence: number;
}

interface ClassificationResult {
  predictions: Prediction[];
  referenceImages: string[];
}

// Mapping of model output indices to dog breeds (This remains, though unused in the modified logic)
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


export async function classifyImage(file: File): Promise<ClassificationResult> {
  try {
    console.log('Starting classification process...');
    
    // Create form data with proper file parameter
    const formData = new FormData();
    formData.append('file', file); // Using 'file' to match FastAPI endpoint
    
    // Send request to backend with proper headers
    const response = await fetch('/api/classify', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Classification failed');
    }
    
    const result = await response.json();
    console.log('Classification complete:', result);
    
    return result;
  } catch (error: any) {
    console.error('Classification error:', error);
    throw new Error(error?.message || 'Failed to classify image');
  }
}

// Memory management utility (This remains, though largely irrelevant without TensorFlow)
export function clearTensorMemory() {
  // No TensorFlow cleanup needed here anymore.  This function is retained for consistency, but is effectively a no-op.
}
import io
import os
import asyncio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import logging

# Create necessary directories
os.makedirs('./models/torch_hub', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Common dog breeds for prototype
BREED_CLASSES = [
    'Labrador', 'German Shepherd', 'Golden Retriever',
    'Bulldog', 'Poodle', 'Beagle', 'Rottweiler',
    'Yorkshire Terrier', 'Boxer', 'Dachshund'
]

class DogBreedClassifier:
    def __init__(self):
        logging.info("Initializing DogBreedClassifier...")
        self.device = torch.device('cpu')
        logging.info(f"Using device: {self.device}")
        self.model = None
        
        # Setup transforms for ResNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        logging.info("Image transforms initialized successfully")

    async def load_model(self):
        """Load the model with proper error handling and retry mechanism"""
        if self.model is not None:
            return

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logging.info(f"Loading ResNet18 model (attempt {retry_count + 1}/{max_retries})...")
                
                # Set torch hub directory to ensure we have write permissions
                torch.hub.set_dir('./models/torch_hub')
                
                # Initialize model with pre-trained weights
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                
                # Replace final layer for our classes
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, len(BREED_CLASSES))
                self.model = self.model.to(self.device)
                
                logging.info("Setting model to evaluation mode...")
                self.model.eval()
                
                # Verify model with small input
                logging.info("Verifying model...")
                dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                logging.info("Model loaded and verified successfully")
                return
                
            except Exception as e:
                logging.error(f"Error loading model (attempt {retry_count + 1}): {str(e)}")
                self.model = None
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying in 2 seconds...")
                    await asyncio.sleep(2)
        
        raise Exception(f"Failed to load model after {max_retries} attempts")

    async def preprocess_image(self, image_bytes):
        """Preprocess image for model input"""
        try:
            logging.info("Preprocessing image...")
            image = Image.open(io.BytesIO(image_bytes))
            if not image:
                raise ValueError("Failed to open image")
            
            image = image.convert('RGB')
            processed = self.transform(image)
            return processed.unsqueeze(0).to(self.device)
            
        except Exception as e:
            logging.error(f"Error in image preprocessing: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    async def classify_image(self, image_bytes):
        """Classify an image and return predictions"""
        try:
            logging.info("Starting image classification...")
            if self.model is None:
                logging.info("Model not loaded, loading now...")
                await self.load_model()
            
            input_tensor = await self.preprocess_image(image_bytes)
            logging.info(f"Input tensor shape: {input_tensor.shape}")
            
            # Get predictions
            logging.info("Running inference...")
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top predictions
            logging.info("Computing top predictions...")
            top_k = 3
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'breed': BREED_CLASSES[idx],
                    'confidence': float(prob)
                })
            
            logging.info(f"Generated predictions: {predictions}")
            
            # Generate reference images
            reference_images = [
                f"https://source.unsplash.com/featured/?{pred['breed'].replace(' ', '+')},dog"
                for pred in predictions
            ]
            
            return {
                'predictions': predictions,
                'referenceImages': reference_images
            }
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            raise Exception(f"Failed to classify image: {str(e)}")

# Create singleton instance
classifier = DogBreedClassifier()
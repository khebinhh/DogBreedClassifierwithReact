import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import requests
from tqdm import tqdm
import tarfile

# Stanford Dogs Dataset classes (simplified for prototype)
BREED_CLASSES = [
    'Labrador', 'German Shepherd', 'Golden Retriever', 'Bulldog', 
    'Beagle', 'Poodle', 'Rottweiler', 'Yorkshire Terrier', 
    'Boxer', 'Dachshund'
]

MODEL_PATH = "models/dog_classifier.pth"

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

class DogBreedClassifier:
    def __init__(self):
        print("Initializing DogBreedClassifier...")
        print(f"PyTorch version: {torch.__version__}")
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        self.model = None
        
        # Setup image transforms for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
        print("Image transforms initialized successfully")

    async def load_model(self):
        """Load the model with proper error handling"""
        if self.model is not None:
            return

        try:
            print("Loading ResNet model...")
            # Initialize model with pre-trained weights
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Replace final layer for our number of classes
            num_features = base_model.fc.in_features
            base_model.fc = nn.Linear(num_features, len(BREED_CLASSES))
            
            self.model = base_model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Verify model
            print("Verifying model...")
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("Model loaded and verified successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None
            raise

    async def preprocess_image(self, image_bytes):
        try:
            # Open and validate image
            image = Image.open(io.BytesIO(image_bytes))
            if not image:
                raise ValueError("Failed to open image")
            
            # Convert to RGB
            image = image.convert('RGB')
            
            # Apply transforms
            processed = self.transform(image)
            return processed.unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    async def classify_image(self, image_bytes):
        try:
            print("Starting image classification...")
            if self.model is None:
                print("Model not loaded, loading now...")
                await self.load_model()
            
            print("Preprocessing image...")
            input_tensor = await self.preprocess_image(image_bytes)
            print(f"Input tensor shape: {input_tensor.shape}")
            
            # Get predictions
            print("Running inference...")
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top predictions
            print("Computing top predictions...")
            top_k = 3
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'breed': BREED_CLASSES[idx],
                    'confidence': float(prob)
                })
            
            print(f"Generated predictions: {predictions}")
            
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
            print(f"Classification error: {str(e)}")
            raise Exception(f"Failed to classify image: {str(e)}")

# Create singleton instance
classifier = DogBreedClassifier()
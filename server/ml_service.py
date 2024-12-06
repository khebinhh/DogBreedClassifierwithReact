import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np

# Dog breed classes (same as before to maintain compatibility)
BREED_CLASSES = [
    'Labrador', 'Golden Retriever', 'German Shepherd', 'Bulldog', 'Poodle',
    'Beagle', 'Rottweiler', 'Yorkshire Terrier', 'Boxer', 'Dachshund'
]

class DogBreedClassifier:
    def __init__(self):
        self.device = torch.device('cpu')  # Force CPU usage
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def load_model(self):
        if self.model is None:
            print("Loading ResNet50 model...")
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.eval()
            self.model.to(self.device)
            print("Model loaded successfully")

    async def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    async def classify_image(self, image_bytes):
        try:
            await self.load_model()
            
            # Preprocess image
            input_tensor = await self.preprocess_image(image_bytes)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=3)
            
            # Convert to list of predictions
            predictions = [
                {
                    'breed': BREED_CLASSES[idx % len(BREED_CLASSES)],
                    'confidence': float(prob)
                }
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())
            ]
            
            # Generate reference images
            reference_images = [
                f"https://source.unsplash.com/featured/?{pred['breed'].lower()},dog"
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

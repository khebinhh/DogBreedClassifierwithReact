import io
import os
import asyncio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import logging
import urllib.request
import tarfile
from tqdm import tqdm

# Create necessary directories
os.makedirs('./models/torch_hub', exist_ok=True)
os.makedirs('./data/stanford-dogs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def download_stanford_dogs():
    """Download Stanford Dogs Dataset"""
    dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    
    print("Downloading Stanford Dogs Dataset...")
    for url in [dataset_url, annotations_url]:
        filename = os.path.basename(url)
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
        
        print(f"Extracting {filename}...")
        with tarfile.open(filename) as tar:
            tar.extractall(path="./data/stanford-dogs")
    
    print("Dataset download and extraction complete")

# Stanford Dogs Dataset breeds
BREED_CLASSES = [
    'Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita', 'Alaskan_malamute',
    'American_bulldog', 'American_pit_bull_terrier', 'American_staffordshire_terrier',
    'Australian_cattle_dog', 'Australian_shepherd', 'Australian_terrier', 'Basenji',
    'Basset_hound', 'Beagle', 'Bedlington_terrier', 'Bernese_mountain_dog',
    'Black-and-tan_coonhound', 'Blenheim_spaniel', 'Bloodhound', 'Bluetick_coonhound',
    'Border_collie', 'Border_terrier', 'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres',
    'Boxer', 'Boykin_spaniel', 'Briard', 'Brittany_spaniel', 'Brussels_griffon',
    'Bull_terrier', 'Bulldog', 'Bullmastiff', 'Cairn_terrier', 'Cardigan_welsh_corgi',
    'Cavalier_king_charles_spaniel', 'Chesapeake_bay_retriever', 'Chihuahua',
    'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow', 'Clumber_spaniel',
    'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian',
    'Dandie_dinmont_terrier', 'Doberman_pinscher', 'English_foxhound',
    'English_setter', 'English_springer_spaniel', 'English_toy_spaniel',
    'Entlebucher_mountain_dog', 'Field_spaniel', 'Finnish_spitz', 'Flat-coated_retriever',
    'French_bulldog', 'German_shepherd', 'German_shorthaired_pointer',
    'German_wirehaired_pointer', 'Giant_schnauzer', 'Golden_retriever', 'Gordon_setter',
    'Great_dane', 'Great_pyrenees', 'Greater_swiss_mountain_dog', 'Groenendael',
    'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel',
    'Irish_wolfhound', 'Italian_greyhound', 'Japanese_chin', 'Japanese_spaniel',
    'Keeshond', 'Kerry_blue_terrier', 'Komondor', 'Kuvasz', 'Labrador_retriever',
    'Lakeland_terrier', 'Leonberger', 'Lhasa_apso', 'Maltese', 'Manchester_terrier',
    'Mastiff', 'Mexican_hairless', 'Miniature_pinscher', 'Miniature_poodle',
    'Miniature_schnauzer', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound',
    'Norwich_terrier', 'Old_english_sheepdog', 'Otterhound', 'Papillon', 'Pekinese',
    'Pembroke_welsh_corgi', 'Pomeranian', 'Pug', 'Redbone_coonhound',
    'Rhodesian_ridgeback', 'Rottweiler', 'Saint_bernard', 'Saluki', 'Samoyed',
    'Schipperke', 'Scottish_deerhound', 'Scottish_terrier', 'Sealyham_terrier',
    'Shetland_sheepdog', 'Shiba_inu', 'Shih-tzu', 'Siberian_husky', 'Silky_terrier',
    'Soft-coated_wheaten_terrier', 'Staffordshire_bull_terrier', 'Standard_poodle',
    'Standard_schnauzer', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier',
    'Toy_poodle', 'Toy_terrier', 'Vizsla', 'Weimaraner', 'Welsh_springer_spaniel',
    'West_highland_white_terrier', 'Whippet', 'Wire-haired_fox_terrier',
    'Yorkshire_terrier'
]

class DogBreedClassifier:
    def __init__(self):
        logging.info("Initializing DogBreedClassifier...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Setup transforms for training with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logging.info("Image transforms initialized successfully")

    async def load_model(self):
        """Load the model with proper error handling"""
        if self.model is not None:
            return

        try:
            logging.info("Loading ResNet50 model...")
            # Initialize model with pre-trained weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # Replace final layers for better feature extraction
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(BREED_CLASSES))
            )
            self.model = self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            logging.info("Model loaded and initialized successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self.model = None
            raise

    async def train_model(self):
        """Train the model on Stanford Dogs Dataset"""
        if self.model is None:
            await self.load_model()
            
        # Training parameters
        num_epochs = 50
        batch_size = 32
        
        # Create data loaders
        train_dataset = ImageFolder('data/stanford-dogs/Images', transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        logging.info("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.3f}, Accuracy: {accuracy:.2f}%')
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), f'models/model_epoch_{epoch+1}.pth')
        
        logging.info("Training completed")
        torch.save(self.model.state_dict(), 'models/final_model.pth')

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

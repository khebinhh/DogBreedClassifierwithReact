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

# Stanford Dogs Dataset classes (all 120 breeds)
BREED_CLASSES = [
    'n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog',
    'n02086079-Pekinese', 'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel',
    'n02086910-papillon', 'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback',
    'n02088094-Afghan_hound', 'n02088238-Basset', 'n02088364-Beagle',
    'n02088466-Bloodhound', 'n02088632-Bluetick', 'n02089078-Black-and-tan_coonhound',
    'n02089867-Walker_hound', 'n02089973-English_foxhound', 'n02090379-Redbone',
    'n02090622-Borzoi', 'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound',
    'n02091134-Whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound',
    'n02091635-Otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound',
    'n02092339-Weimaraner', 'n02093256-Staffordshire_bullterrier',
    'n02093428-American_Staffordshire_terrier', 'n02093647-Bedlington_terrier',
    'n02093754-Border_terrier', 'n02093859-Kerry_blue_terrier',
    'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier',
    'n02094258-Norwich_terrier', 'n02094433-Yorkshire_terrier',
    'n02095314-Wire-haired_fox_terrier', 'n02095570-Lakeland_terrier',
    'n02095889-Sealyham_terrier', 'n02096051-Airedale',
    'n02096177-Cairn', 'n02096294-Australian_terrier',
    'n02096437-Dandie_Dinmont', 'n02096585-Boston_bull',
    'n02097047-Miniature_schnauzer', 'n02097130-Giant_schnauzer',
    'n02097209-Standard_schnauzer', 'n02097298-Scotch_terrier',
    'n02097474-Tibetan_terrier', 'n02097658-Silky_terrier',
    'n02098105-Soft-coated_wheaten_terrier', 'n02098286-West_Highland_white_terrier',
    'n02098413-Lhasa', 'n02099267-Flat-coated_retriever',
    'n02099429-Curly-coated_retriever', 'n02099601-Golden_retriever',
    'n02099712-Labrador_retriever', 'n02099849-Chesapeake_Bay_retriever',
    'n02100236-German_short-haired_pointer', 'n02100583-Vizsla',
    'n02100735-English_setter', 'n02100877-Irish_setter',
    'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel',
    'n02101556-Clumber', 'n02102040-English_springer',
    'n02102177-Welsh_springer_spaniel', 'n02102318-Cocker_spaniel',
    'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel',
    'n02104029-Kuvasz', 'n02104365-Schipperke', 'n02105056-Groenendael',
    'n02105162-Malinois', 'n02105251-Briard', 'n02105412-Kelpie',
    'n02105505-Komondor', 'n02105641-Old_English_sheepdog',
    'n02105855-Shetland_sheepdog', 'n02106030-Collie',
    'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres',
    'n02106550-Rottweiler', 'n02106662-German_shepherd',
    'n02107142-Doberman', 'n02107312-Miniature_pinscher',
    'n02107574-Greater_Swiss_Mountain_dog', 'n02107683-Bernese_mountain_dog',
    'n02107908-Appenzeller', 'n02108000-EntleBucher',
    'n02108089-Boxer', 'n02108422-Bull_mastiff', 'n02108551-Tibetan_mastiff',
    'n02108915-French_bulldog', 'n02109047-Great_Dane',
    'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog',
    'n02110063-Malamute', 'n02110185-Siberian_husky',
    'n02110341-Dalmatian', 'n02110627-Affenpinscher',
    'n02110806-Basenji', 'n02110958-Pug', 'n02111129-Leonberg',
    'n02111277-Newfoundland', 'n02111500-Great_Pyrenees',
    'n02111889-Samoyed', 'n02112018-Pomeranian', 'n02112137-Chow',
    'n02112350-Keeshond', 'n02112706-Brabancon_griffon',
    'n02113023-Pembroke', 'n02113186-Cardigan', 'n02113624-Toy_poodle',
    'n02113712-Miniature_poodle', 'n02113799-Standard_poodle',
    'n02113978-Mexican_hairless', 'n02114367-Timber_wolf',
    'n02114548-White_wolf', 'n02114712-Red_wolf', 'n02114855-Coyote',
    'n02115641-Dingo', 'n02115913-Dhole', 'n02116738-African_hunting_dog'
]

DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
MODEL_PATH = "models/dog_classifier.pth"

def download_and_extract_dataset(dataset_path="dataset"):
    """Download and extract Stanford Dogs Dataset"""
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
    # Download dataset
    if not os.path.exists(f"{dataset_path}/images.tar"):
        print("Downloading Stanford Dogs Dataset...")
        response = requests.get(DATASET_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(f"{dataset_path}/images.tar", 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(len(data))
        
        # Extract dataset
        print("Extracting dataset...")
        with tarfile.open(f"{dataset_path}/images.tar") as tar:
            tar.extractall(dataset_path)

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final fully connected layers with a more robust classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

    @staticmethod
    def validate_breed_classes():
        """Validate that all breed classes are properly formatted"""
        try:
            for breed in BREED_CLASSES:
                if not breed.startswith('n') or not '-' in breed:
                    raise ValueError(f"Invalid breed format: {breed}")
            print(f"Successfully validated {len(BREED_CLASSES)} breed classes")
            return True
        except Exception as e:
            print(f"Error validating breed classes: {str(e)}")
            return False

class DogBreedClassifier:
    def __init__(self):
        print("Initializing DogBreedClassifier...")
        self.device = torch.device('cpu')  # Force CPU usage for Replit
        self.model = None
        print("Setting up image transforms...")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def load_model(self):
        if self.model is None:
            try:
                print("Loading custom ResNet model...")
                self.model = CustomResNet(len(BREED_CLASSES))
                
                # Load trained weights if they exist
                if os.path.exists(MODEL_PATH):
                    print(f"Loading trained weights from {MODEL_PATH}")
                    self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                else:
                    print("No trained weights found. Using pre-trained model.")
                    await self.train_model()
                
                self.model.eval()
                self.model.to(self.device)
                print(f"Model loaded successfully on device: {self.device}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise Exception(f"Failed to load model: {str(e)}")
                
    async def train_model(self, epochs=10, batch_size=32):
        """Train the model on Stanford Dogs Dataset"""
        try:
            print("Starting model training...")
            
            # Prepare dataset
            dataset_path = "dataset/Images"
            if not os.path.exists(dataset_path):
                download_and_extract_dataset()
            
            # Create dataset and dataloaders
            full_dataset = ImageFolder(dataset_path, transform=self.train_transform)
            
            # Split dataset
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model if not already initialized
            if self.model is None:
                self.model = CustomResNet(len(BREED_CLASSES))
                self.model.to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            best_val_acc = 0
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                val_acc = 100. * correct / total
                print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss/len(val_loader):.4f}, Val Acc = {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    torch.save(self.model.state_dict(), MODEL_PATH)
                    print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            
            print("Training completed!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise Exception(f"Failed to train model: {str(e)}")

    async def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

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
            top_probs, top_indices = torch.topk(probabilities, k=3)
            
            # Convert to list of predictions
            predictions = [
                {
                    'breed': BREED_CLASSES[idx % len(BREED_CLASSES)],
                    'confidence': float(prob)
                }
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())
            ]
            
            # Generate reference images with proper breed name formatting
            reference_images = []
            for pred in predictions:
                # Extract the actual breed name from the class ID (remove the initial n* prefix)
                breed_name = pred['breed'].split('-', 1)[1].replace('_', ' ').title()
                reference_images.append(
                    f"https://source.unsplash.com/featured/?{breed_name},dog"
                )
                print(f"Generated reference image for breed: {breed_name}")
            
            return {
                'predictions': predictions,
                'referenceImages': reference_images
            }
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            raise Exception(f"Failed to classify image: {str(e)}")

# Create singleton instance
classifier = DogBreedClassifier()

from utils.card import Card
import random
import time
from utils.agent import CardRecognizer
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

class Deck:
    def __init__(self):
        self.cards = []
        # Create a single instance of CardRecognizer to be reused
        self.recognizer = CardRecognizer(device="cpu")
        self.reset()
        
    def reset(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        start_time = time.time()
        # Define image transformation with grayscale conversion
        transform = transforms.Compose([
            transforms.Resize((134, 134)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        
        for card in self.cards:
            # Get image path
            image_path = card.get_image_path()
            if os.path.exists(image_path):
                # Load and transform image to tensor
                image = Image.open(image_path).convert('L')
                image_tensor = transform(image)
                # Pass tensor to recognizer
                card.predicted_category, card.predicted_suit = self.recognizer.classify_card(image_tensor)
            else:
                print(f"Image not found: {image_path}")
        elapsed_time = time.time() - start_time
        print(f"Time taken to process the dataset one by one: {elapsed_time:.2f} seconds, {elapsed_time/len(self.cards):.2f} seconds per image, total {len(self.cards)} images")
        random.shuffle(self.cards)
        
    def deal(self):
        if not self.cards:
            self.reset()
        return self.cards.pop()
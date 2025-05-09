from utils.card import Card
import random
class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
        
    def reset(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.cards)
        
    def deal(self):
        if not self.cards:
            self.reset()
        return self.cards.pop()
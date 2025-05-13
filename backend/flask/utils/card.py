import random
class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.number = random.randint(1, 5)
        self.predicted_category, self.predicted_suit = None, None
        
    def get_numeric_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11
        else:
            return int(self.value)
    
    def get_predicted_card_value(self):
        score = 0
        value_names = {
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
            'jack': 'J',
            'queen': 'Q',
            'king': 'K',
            'ace': 'A'
        }
        
        # Translate the predicted category using the value_names dictionary
        if self.predicted_category in value_names:
            categori = value_names[self.predicted_category]
        else:
            ValueError(f"Invalid predicted category: {self.predicted_category}")
            
        # Score from the predicted category
        if categori in ['J', 'Q', 'K']:
            score = 10
        elif categori == 'A':
            score = 11
        else:
            score = int(self.value)
        
        # Score from the predicted suit
        if self.predicted_suit == 'hearts':
            score += 2
        elif self.predicted_suit == 'diamonds':
            score += 1
        elif self.predicted_suit == 'spades':
            score -= 2
            
        aces = 1 if self.value == 'A' else 0

        while score > 21 and aces > 0:
            score -= 10
            aces -= 1
        
        return score

        
    
    def get_image_path(self):
        value_names = {
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine',
            '10': 'ten',
            'J': 'jack',
            'Q': 'queen',
            'K': 'king',
            'A': 'ace'
        }
        
        value_name = value_names.get(self.value, self.value)
        
        card_name = f"{value_name} of {self.suit}"
        
        
        return f"/static/data/test/{card_name}/{self.number}.jpg"
            
    def to_dict(self):
        return {
            'suit': self.suit,
            'value': self.value,
            'numeric_value': self.get_numeric_value(),
            'image_path': self.get_image_path(),
            'predicted_category': self.predicted_category,
            'predicted_suit': self.predicted_suit
        }
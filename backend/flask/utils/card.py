import random
class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.number = random.randint(1, 5)
        
    def get_numeric_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11
        else:
            return int(self.value)
    
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
            'image_path': self.get_image_path()
        }
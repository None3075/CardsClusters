from utils.card import Card
from utils.deck import Deck
import random
class Game:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.message = ""
        self.turn = "player"
        self.dealer_stand = False
        
    def start_game(self):
        self.deck.reset()
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]
        self.game_over = False
        self.message = "Game started. Player's turn."
        self.turn = "player"
        self.dealer_stand = False
        return self.get_game_state()
    
    def calculate_score(self, hand):
        score = sum(card.get_numeric_value() for card in hand)
        aces = sum(1 for card in hand if card.value == 'A')
        while score > 21 and aces > 0:
            score -= 10
            aces -= 1
        return score
    
    def player_hit(self):
        if self.game_over or self.turn != "player":
            return self.get_game_state()
        
        self.player_hand.append(self.deck.deal())
        player_score = self.calculate_score(self.player_hand)
        
        if player_score > 21:
            self.game_over = True
            self.message = "Bust! Dealer wins."
            return self.get_game_state()
        
        self.message = "Player hit. Dealer's turn."
        self.turn = "dealer"
        return self.dealer_turn()
    
    def player_stand(self):
        if self.game_over or self.turn != "player":
            return self.get_game_state()
        
        # if player stand and dealer has stand end the game
        if self.dealer_stand:
            self.game_over = True
            player_score = self.calculate_score(self.player_hand)
            dealer_score = self.calculate_score(self.dealer_hand)
            if dealer_score > 21 or player_score > dealer_score:
                self.message = "Player wins!"
            elif player_score < dealer_score:
                self.message = "Dealer wins!"
            else:
                self.message = "It's a tie!"
            return self.get_game_state()
            
        self.message = "Player stands. Dealer's turn."
        self.turn = "dealer"
        return self.dealer_turn()
    
    def dealer_turn(self):
        if self.game_over:
            return self.get_game_state()
        
        # Dealer's logic: hit on 16 or less, stand on 17 or more
        dealer_score = self.calculate_score(self.dealer_hand)
        
        if dealer_score < 17:
            self.dealer_hand.append(self.deck.deal())
            self.message = "Dealer hits."
            
            dealer_score = self.calculate_score(self.dealer_hand)
            if dealer_score > 21:
                self.game_over = True
                self.message = "Dealer busts! You win!"
                return self.get_game_state()
        else:
            self.message = "Dealer stands."
            self.dealer_stand = True
        
        # Switch turn back to player
        self.turn = "player"
        self.message += " Player's turn."
        return self.get_game_state()
    
    def get_game_state(self):
        dealer_hand_to_show = self.dealer_hand
        if not self.game_over and self.turn == "player" and len(self.dealer_hand) > 1:
            dealer_hand_to_show = [self.dealer_hand[0]]
        
        return {
            "playerHand": [card.to_dict() for card in self.player_hand],
            "playerScore": self.calculate_score(self.player_hand),
            "dealerHand": [card.to_dict() for card in dealer_hand_to_show],
            "dealerScore": self.calculate_score(dealer_hand_to_show) if not self.game_over else self.calculate_score(self.dealer_hand),
            "gameOver": self.game_over,
            "message": self.message,
            "turn": self.turn,
            "dealerHandComplete": self.game_over or self.turn == "dealer",
            "dealerHandCount": len(self.dealer_hand)
        }
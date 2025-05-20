from utils.card import Card
from utils.deck import Deck
import random
from utils.utils.decision_maker import DecisionMaker

class Game:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.message = ""
        self.turn = "player"
        self.dealer_stand = False
        self.dealer_agent = DecisionMaker()
        
    def start_game(self):
        self.deck.reset()
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]
        self.game_over = False
        self.message = "Game started. Player's turn."
        self.turn = "player"
        self.dealer_stand = False
        
        # Check if the game is over at the start because player or dealer drawed more than 21
        player_score = self.calculate_score(self.player_hand)
        dealer_score = self.calculate_score(self.dealer_hand)
        if player_score > 21:
            self.game_over = True
            self.message = "Bust! Dealer wins."
            return self.get_game_state()
        elif dealer_score > 21:
            self.game_over = True
            self.message = "Dealer busts! You win!"
            return self.get_game_state()
        else:
            return self.get_game_state()
    
    def calculate_score(self, hand):
        score = sum(card.get_numeric_value() for card in hand)
        
        for card in hand:
            if card.suit == 'hearts':
                score += 2
            elif card.suit == 'diamonds':
                score += 1
            elif card.suit == 'spades':
                score -= 2
            
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
        player_visible_card_value = self.player_hand[0].get_predicted_card_value()
        # the state for dealer agent is dealer_handvalue, player_firstcardvalue, usable_aces (usable_aces represents the number of ace cards that are being counted with their higher value)
        state = (
            dealer_score,
            player_visible_card_value,
            self.calculate_usable_aces(self.dealer_hand)
            )
        
        dealer_action = self.dealer_agent.decide(state)
        
        print(f"Dealer's action: {dealer_action}")
        if dealer_action == 1:  # Hit
            self.dealer_hand.append(self.deck.deal())
            self.message = "Dealer hits."
            
            dealer_score = self.calculate_score(self.dealer_hand)
            if dealer_score > 21:
                self.game_over = True
                self.message = "Dealer busts! You win!"
                return self.get_game_state()
        elif dealer_action == 0:  # Stand
            self.message = "Dealer stands."
            self.dealer_stand = True
        else:
            self.message = "Invalid action. Dealer's turn."
            self.dealer_stand = True
        
        # Switch turn back to player
        self.turn = "player"
        self.message += " Player's turn."
        return self.get_game_state()
    
    # usable_aces represents the number of ace cards that are being counted with their higher value
    def calculate_usable_aces(self, hand):
        # Count total aces in hand
        total_aces = sum(1 for card in hand if card.value == 'A')
        
        if total_aces == 0:
            return 0
        
        # Calculate initial score counting all aces as 11
        score = 0
        for card in hand:
            if card.value == 'A':
                score += 11  # Initially count all aces as 11
            else:
                score += card.get_numeric_value()
                
            # Add suit-based adjustments
            if card.suit == 'hearts':
                score += 2
            elif card.suit == 'diamonds':
                score += 1
            elif card.suit == 'spades':
                score -= 2
        
        # Count how many aces need to be reduced to avoid busting
        reduced_aces = 0
        while score > 21 and reduced_aces < total_aces:
            score -= 10  # Convert one ace from 11 to 1
            reduced_aces += 1
        
        # Return number of aces still counted as 11
        return total_aces - reduced_aces
    
    
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
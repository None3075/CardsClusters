from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os
from utils.card import Card
from utils.deck import Deck
from utils.game import Game

app = Flask(__name__, static_folder="static")
CORS(app)

game = Game()

@app.route("/api/game/start", methods=["POST"])
def start_game():
    return jsonify(game.start_game())

@app.route("/api/game/hit", methods=["POST"])
def player_hit():
    if game.turn != "player":
        return jsonify({"error": "Not player's turn"}), 400
    return jsonify(game.player_hit())

@app.route("/api/game/stand", methods=["POST"])
def player_stand():
    if game.turn != "player":
        return jsonify({"error": "Not player's turn"}), 400
    return jsonify(game.player_stand())

@app.route("/api/game/dealer", methods=["POST"])
def dealer_action():
    if game.turn != "dealer":
        return jsonify({"error": "Not dealer's turn"}), 400
    return jsonify(game.dealer_turn())

@app.route("/api/game/state", methods=["GET"])
def game_state():
    return jsonify(game.get_game_state())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
import { useState } from 'react';
import axios from 'axios';
import { motion } from "framer-motion";
import useSound from 'use-sound';
import dealSound from './sounds/card-deal.mp3';
import winSound from './sounds/win.mp3';
import loseSound from './sounds/lose.mp3';

function App() {
  const [gameState, setGameState] = useState({
    playerHand: [],
    dealerHand: [],
    playerScore: 0,
    dealerScore: 0,
    gameOver: false,
    message: ""
  });
  
  const [loading, setLoading] = useState(false);
  const API_URL = "http://localhost:5000/api/game"; // TODO IPS CHANGE

  const [playDeal] = useSound(dealSound);
  const [playWin] = useSound(winSound);
  const [playLose] = useSound(loseSound);

  const startGame = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/start`);
      setGameState(response.data);
      playDeal();
    } catch (error) {
      console.error("Error starting game:", error);
    } finally {
      setLoading(false);
    }
  };

  const hit = async () => {
    if (gameState.gameOver) return;
    playDeal();
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/hit`);
      setGameState(response.data);
    } catch (error) {
      console.error("Error hitting:", error);
    } finally {
      setLoading(false);
    }
  };

  const stand = async () => {
    if (gameState.gameOver) return;
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/stand`);
      const result = response.data;
      setGameState(result);
      
      // Probablemente esto no se deberia de hacer asi but who cares
      if (result.message && result.message.includes("You")) {
        playWin();
      } else if (result.message && (result.message.includes("Bust") || 
                 result.message.includes("Dealer wins"))) {
        playLose();
      }
    } catch (error) {
      console.error("Error standing:", error);
    } finally {
      setLoading(false);
    }
  };

  const renderCard = (card, index) => {
    const imageUrl = `http://localhost:5000${card.image_path}`; // TODO IPS CHANGE
    
    return (
      <motion.div
        key={`${card.value}-${card.suit}`}
        className="relative w-24 h-36 m-2 rounded-lg shadow-lg"
        initial={{ opacity: 0, y: 50, rotateY: 180 }}
        animate={{ opacity: 1, y: 0, rotateY: 0 }}
        transition={{ delay: index * 0.2, duration: 0.5 }}
        whileHover={{ y: -10, transition: { duration: 0.2 } }}
      >
        <img 
          src={imageUrl} 
          alt={`${card.value} of ${card.suit}`} 
          className="w-full h-full rounded-lg border-2 border-white"
        />
      </motion.div>
    );
  };

  const renderHiddenCard = () => {
    return (
      <div className="relative w-24 h-36 m-2 rounded-lg shadow-lg">
        <div className="w-full h-full rounded-lg bg-red-700 border-2 border-white flex items-center justify-center">
          <div className="bg-red-800 w-4/5 h-4/5 rounded-lg flex items-center justify-center">
            <div className="bg-red-600 w-3/5 h-3/5 rounded-md transform rotate-45"></div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-green-800 text-white flex flex-col">
      <header className="bg-gradient-to-r from-yellow-700 to-yellow-600 p-4 shadow-lg">
        <h1 className="text-4xl font-bold text-center font-serif tracking-wider">BlackJack</h1>
      </header>
      
      <div className="flex-1 flex flex-col p-6 max-w-5xl mx-auto w-full">
        {loading && (
          <motion.div 
            className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 border-4 border-t-transparent border-yellow-400 rounded-full animate-spin"></div>
              <p className="text-yellow-400 mt-3 font-bold">Shuffling cards...</p>
            </div>
          </motion.div>
        )}
        {gameState.playerHand.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full space-y-8">
            <h2 className="text-3xl font-bold text-yellow-300 drop-shadow-lg">Welcome to Blackjack!</h2>
            <button 
              className="bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 
                        text-black font-bold py-3 px-6 rounded-lg shadow-lg transform hover:scale-105 transition-all
                        border-2 border-yellow-300"
              onClick={startGame}
              disabled={loading}
            >
              {loading ? 'Shuffling Cards...' : 'Deal Cards'}
            </button>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <h2 className="text-2xl font-bold mb-2 text-yellow-300">
                Dealer's Hand {!gameState.gameOver && <span className="text-white font-normal text-lg">(showing {gameState.dealerHand[0].numeric_value})</span>}
              </h2>
              <div className="flex flex-wrap">
                {gameState.dealerHand.map((card, index) => renderCard(card, index))}
                {/* Add hidden card if dealer hand is not complete */}
                {!gameState.dealerHandComplete && gameState.dealerHand.length === 1 && renderHiddenCard()}
              </div>
              {gameState.gameOver && (
                <div className="mt-2 text-xl font-semibold text-yellow-200">Score: {gameState.dealerScore}</div>
              )}
            </div>
            
            {gameState.message && (
              <motion.div 
                className="my-4 p-3 bg-opacity-50 bg-black rounded-lg text-center"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
              >
                <p className="text-2xl font-bold text-yellow-300">{gameState.message}</p>
              </motion.div>
            )}
            
            <div className="mt-auto">
              <h2 className="text-2xl font-bold mb-2 text-yellow-300">
                Your Hand <span className="text-white font-normal text-lg">({gameState.playerScore})</span>
              </h2>
              <div className="flex flex-wrap">
                {gameState.playerHand.map((card, index) => renderCard(card, index))}
              </div>
              
              <motion.div 
                className="mt-2 text-xl font-semibold text-yellow-200"
                key={gameState.playerScore}
                initial={{ opacity: 0, scale: 1.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                Score: {gameState.playerScore}
              </motion.div>
              
              <div className="flex space-x-4 mt-6">
                <motion.button 
                  className={`px-6 py-2 rounded-lg font-bold shadow-lg transform hover:scale-105 transition-all
                             ${loading || gameState.gameOver 
                               ? 'bg-gray-600 cursor-not-allowed' 
                               : 'bg-red-600 hover:bg-red-500 text-white'}`}
                  onClick={hit} 
                  disabled={loading || gameState.gameOver}
                  whileTap={{ scale: 0.95 }}
                  whileHover={loading || gameState.gameOver ? {} : { scale: 1.05 }}
                >
                  Hit
                </motion.button>
                
                <motion.button 
                  className={`px-6 py-2 rounded-lg font-bold shadow-lg transform hover:scale-105 transition-all
                             ${loading || gameState.gameOver 
                               ? 'bg-gray-600 cursor-not-allowed' 
                               : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                  onClick={stand} 
                  disabled={loading || gameState.gameOver}
                  whileTap={{ scale: 0.95 }}
                  whileHover={loading || gameState.gameOver ? {} : { scale: 1.05 }}
                >
                  Stand
                </motion.button>
                
                {gameState.gameOver && (
                  <motion.button 
                    className="bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 
                              text-black font-bold px-6 py-2 rounded-lg shadow-lg border-2 border-yellow-300"
                    onClick={startGame}
                    disabled={loading}
                    whileTap={{ scale: 0.95 }}
                    whileHover={{ scale: 1.05 }}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    New Game
                  </motion.button>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
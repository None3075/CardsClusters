import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from "framer-motion";
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
    message: "",
    turn: "",
    dealerHandComplete: false
  });
  
  const [loading, setLoading] = useState(false);
  const API_URL = "http://localhost:5000/api/game"; // TODO IPS CHANGE

  const [playDeal] = useSound(dealSound);
  const [playWin] = useSound(winSound);
  const [playLose] = useSound(loseSound);

  // When the game state changes, check if we need to fetch the latest state
  useEffect(() => {
    // Poll for game state periodically to ensure UI is in sync with backend
    const pollInterval = setInterval(async () => {
      if (gameState.playerHand.length > 0 && !loading) {
        await refreshGameState();
      }
    }, 2000);
    
    return () => clearInterval(pollInterval);
  }, []);

  // Separate effect for dealer's turn
  useEffect(() => {
    if (gameState.turn === "dealer" && !gameState.gameOver && !loading) {
      const dealerAction = async () => {
        setLoading(true);
        try {
          const response = await axios.post(`${API_URL}/dealer`);
          setGameState(response.data);
          playDeal();
          
          if (response.data.gameOver) {
            if (response.data.message && response.data.message.includes("You win")) {
              playWin();
            } else if (response.data.message && (response.data.message.includes("Bust") || 
                     response.data.message.includes("Dealer wins"))) {
              playLose();
            }
          }
        } catch (error) {
          console.error("Error during dealer turn:", error);
        } finally {
          setLoading(false);
        }
      };
      
      // Add a delay before dealer acts for better visual effect
      const timerId = setTimeout(() => {
        dealerAction();
      }, 1000);
      
      return () => clearTimeout(timerId);
    }
  }, [gameState.turn, gameState.gameOver, loading]);

  // Get the latest game state
  const refreshGameState = async () => {
    try {
      const response = await axios.get(`${API_URL}/state`);
      setGameState(response.data);
    } catch (error) {
      console.error("Error refreshing game state:", error);
    }
  };

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
    if (gameState.gameOver || gameState.turn !== "player" || loading) return;
    
    setLoading(true);
    try {
      playDeal();
      const response = await axios.post(`${API_URL}/hit`);
      setGameState(response.data);
      
      if (response.data.message && response.data.message.includes("Bust")) {
        playLose();
      }
    } catch (error) {
      console.error("Error hitting:", error);
    } finally {
      setLoading(false);
    }
  };

  const stand = async () => {
    if (gameState.gameOver || gameState.turn !== "player" || loading) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/stand`);
      setGameState(response.data);
    } catch (error) {
      console.error("Error standing:", error);
    } finally {
      setLoading(false);
    }
  };

  const renderCard = (card, index, isNewCard = false) => {
    const imageUrl = `http://localhost:5000${card.image_path}`; // TODO IPS CHANGE
    
    return (
      <motion.div
        key={`${card.value}-${card.suit}-${index}`}
        className={`relative w-24 h-36 m-2 rounded-lg shadow-lg ${isNewCard ? 'border-4 border-yellow-300' : ''}`}
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

  const renderHiddenCard = (index = 0) => {
    return (
      <div key={`hidden-card-${index}`} className="relative w-24 h-36 m-2 rounded-lg shadow-lg">
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
              <p className="text-yellow-400 mt-3 font-bold">
                {gameState.turn === "dealer" ? "Dealer thinking..." : "Processing..."}
              </p>
            </div>
          </motion.div>
        )}
        
        {/* Game Status Bar */}
        {gameState.playerHand.length > 0 && (
          <motion.div 
            className={`mb-4 p-3 rounded-lg text-center ${
              gameState.turn === "player" 
                ? "bg-blue-800 border-2 border-blue-400" 
                : gameState.turn === "dealer"
                  ? "bg-red-800 border-2 border-red-400"
                  : "bg-gray-800"
            }`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            key={gameState.turn + gameState.gameOver}
          >
            <div className="flex items-center justify-between">
              <div className="w-40 text-left">
                {gameState.gameOver ? (
                  <span className="text-yellow-300 font-bold">GAME OVER</span>
                ) : gameState.turn === "player" ? (
                  <span className="text-blue-300 font-bold">YOUR TURN</span>
                ) : (
                  <span className="text-red-300 font-bold">DEALER'S TURN</span>
                )}
              </div>
              
              <div className="flex-1 text-center text-xl font-bold text-yellow-200">
                {gameState.message}
              </div>
              
              <div className="w-40 text-right">
                {gameState.gameOver ? (
                  <motion.button 
                    className="bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 
                              text-black font-bold px-4 py-1 rounded-lg shadow-lg border-2 border-yellow-300"
                    onClick={startGame}
                    whileTap={{ scale: 0.95 }}
                    whileHover={{ scale: 1.05 }}
                    disabled={loading}
                  >
                    New Game
                  </motion.button>
                ) : (
                  <button 
                    className="bg-gray-700 hover:bg-gray-600 px-4 py-1 rounded-lg text-gray-300"
                    onClick={refreshGameState}
                    disabled={loading}
                  >
                    Refresh
                  </button>
                )}
              </div>
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
            <div className={`mb-8 p-4 rounded-lg ${gameState.turn === "dealer" && !gameState.gameOver ? "bg-red-900 bg-opacity-30 border border-red-500" : ""}`}>
              <h2 className="text-2xl font-bold mb-2 text-yellow-300 flex items-center">
                Dealer's Hand 
                {!gameState.gameOver && !gameState.dealerHandComplete && (
                  <span className="text-white font-normal text-lg ml-2">(showing {gameState.dealerHand[0]?.numeric_value})</span>
                )}
                {gameState.turn === "dealer" && !gameState.gameOver && (
                  <span className="ml-auto py-1 px-3 bg-red-800 rounded-full text-sm animate-pulse">
                    Deciding...
                  </span>
                )}
              </h2>
              
              <div className="flex flex-wrap">
                {gameState.dealerHand.map((card, index) => renderCard(card, index))}
                {!gameState.dealerHandComplete && gameState.dealerHandCount > gameState.dealerHand.length && 
                  Array(gameState.dealerHandCount - gameState.dealerHand.length).fill().map((_, index) => 
                    renderHiddenCard(index + gameState.dealerHand.length)
                  )
                }
              </div>
              
              <AnimatePresence>
                {(gameState.gameOver || gameState.dealerHandComplete) && (
                  <motion.div 
                    className="mt-2 text-xl font-semibold text-yellow-200"
                    key={gameState.dealerScore}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    Score: {gameState.dealerScore}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            
            <div className={`mt-auto p-4 rounded-lg ${gameState.turn === "player" && !gameState.gameOver ? "bg-blue-900 bg-opacity-30 border border-blue-500" : ""}`}>
              <h2 className="text-2xl font-bold mb-2 text-yellow-300 flex items-center">
                Your Hand 
                <span className="text-white font-normal text-lg ml-2">({gameState.playerScore})</span>
                {gameState.turn === "player" && !gameState.gameOver && (
                  <span className="ml-auto py-1 px-3 bg-blue-800 rounded-full text-sm animate-pulse">
                    Your Move
                  </span>
                )}
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
                  className={`px-6 py-2 rounded-lg font-bold shadow-lg transform transition-all
                             ${loading || gameState.gameOver || gameState.turn !== "player"
                               ? 'bg-gray-600 cursor-not-allowed opacity-50' 
                               : 'bg-red-600 hover:bg-red-500 text-white'}`}
                  onClick={hit} 
                  disabled={loading || gameState.gameOver || gameState.turn !== "player"}
                  whileTap={{ scale: 0.95 }}
                  whileHover={loading || gameState.gameOver || gameState.turn !== "player" ? {} : { scale: 1.05 }}
                >
                  Hit
                </motion.button>
                
                <motion.button 
                  className={`px-6 py-2 rounded-lg font-bold shadow-lg transform transition-all
                             ${loading || gameState.gameOver || gameState.turn !== "player"
                               ? 'bg-gray-600 cursor-not-allowed opacity-50' 
                               : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                  onClick={stand} 
                  disabled={loading || gameState.gameOver || gameState.turn !== "player"}
                  whileTap={{ scale: 0.95 }}
                  whileHover={loading || gameState.gameOver || gameState.turn !== "player" ? {} : { scale: 1.05 }}
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
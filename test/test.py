import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy 
import math
import time
from mcts_agent import MCTSAgent, Node
def test_backpropagate():
    # Initialize game state
    game = Game()
    game.place_checker(0, 0)  # PLAYER1
    game.place_checker(1, 1)  # PLAYER2
    game.current_player = PLAYER1
    
    # Initialize agent and node
    agent = MCTSAgent(player=PLAYER1)
    root = Node(copy.deepcopy(game), agent=agent)
    
    # Simulate a win for PLAYER1
    simulated_winner = PLAYER1
    agent.backpropagate(root, simulated_winner)
    
    assert root.visits == 1, "Visit count should be incremented."
    assert root.wins == 1, "Win count should be incremented for PLAYER1 win."
    
    # Simulate a loss for PLAYER1 (PLAYER2 win)
    agent.backpropagate(root, PLAYER2)
    
    assert root.visits == 2, "Visit count should be incremented."
    assert root.wins == 0, "Win count should be decremented for PLAYER2 win."
    
    # Simulate a draw
    agent.backpropagate(root, EMPTY)
    
    assert root.visits == 3, "Visit count should be incremented."
    assert root.wins == 0, "Win count should remain unchanged for a draw."
    
    print("Backpropagate method tests passed.")

# Run the test
test_backpropagate()

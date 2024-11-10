import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy 
import math
import time

import logging

# Configure logging for Player 1
player1_logger = logging.getLogger('Player1Logger')
player1_logger.setLevel(logging.DEBUG)  # Set the desired logging level

# Create a file handler for Player 1
player1_handler = logging.FileHandler('player1.log')
player1_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
player1_formatter = logging.Formatter('%(asctime)s - Player1 - %(levelname)s - %(message)s')
player1_handler.setFormatter(player1_formatter)

# Add the handler to the Player 1 logger
player1_logger.addHandler(player1_handler)

# Configure logging for Player 2
player2_logger = logging.getLogger('Player2Logger')
player2_logger.setLevel(logging.DEBUG)  # Set the desired logging level

# Create a file handler for Player 2
player2_handler = logging.FileHandler('player2.log')
player2_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
player2_formatter = logging.Formatter('%(asctime)s - Player2 - %(levelname)s - %(message)s')
player2_handler.setFormatter(player2_formatter)

# Add the handler to the Player 2 logger
player2_logger.addHandler(player2_handler)

# Optionally, prevent loggers from propagating to the root logger
player1_logger.propagate = False
player2_logger.propagate = False


import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy 
import math
import time
import logging  # Ensure logging is imported

class GoatAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.time_limit = 1  # Time limit per move in seconds

        
        # Assign the appropriate logger based on the player
        if self.player == PLAYER1:
            self.logger = player1_logger
        else:
            self.logger = player2_logger

    # given the game state, gets all of the possible moves
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.is_valid_placement(r, c):
                        moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.is_valid_move(r0, c0, r1, c1):
                                        moves.append((r0, c0, r1, c1))
        return moves
        
    def get_best_move(self, game):
        self.simulation_count = 0
        self.max_simulation_depth = 0
        possible_moves = self.get_possible_moves(game)
        
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        enemy = PLAYER2 if self.player == PLAYER1 else PLAYER1

        if current_pieces < NUM_PIECES:
            # Check for a win
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if game.board[row][col] != EMPTY:
                        continue
                    temp = copy.deepcopy(game)
                    try:
                        temp.place_checker(row, col)  # Use the method instead of direct assignment
                    except ValueError:
                        continue  # Skip invalid placements
                    if temp.check_winner() == self.player:
                        self.logger.info("FOUND A WINNER")
                        return [row, col]

        # **Hardcoded Check for Enemy Diagonal Threats**
        blocking_move = self.check_and_block_enemy_diagonal_win(game)
        if blocking_move:
            self.logger.debug(f"Blocking enemy diagonal win by placing at {blocking_move}")
            self.logger.info(f"Blocking enemy diagonal win by placing at {blocking_move}")
            return blocking_move

        # **Hardcoded Check for Enemy Linear Threats**
        linear_blocking_move = self.check_and_block_enemy_linear_win(game)
        if linear_blocking_move:
            self.logger.info(f"Blocking enemy linear win by placing at {linear_blocking_move}")
            self.logger.debug(f"Blocking enemy linear win by placing at {linear_blocking_move}")
            return linear_blocking_move

        # Check for a block
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game.board[row][col] != EMPTY:
                    continue
                temp = copy.deepcopy(game)
                try:
                    temp.current_player = enemy       # Set current player to enemy for win check
                    temp.place_checker(row, col)  # Temporarily place enemy's piece
                except ValueError:
                    continue  # Skip invalid placements
                if temp.check_winner() == enemy:
                    self.logger.info("FOUND A BLOCK")
                    return [row, col]  

        return random.choice(possible_moves)
                        
     


    ## HARDCODE CHECKS
    def check_and_block_enemy_diagonal_win(self, game):
        """
        Scans all 2x2 subsections of the grid to find if the enemy has two diagonally placed pieces.
        For each threat found, evaluates both blocking positions by simulating each and chooses the one that
        prevents the enemy from having an immediate winning move in their next turn.

        Returns:
            move (list): [row, col] if a blocking move is found, else None.
        """
        enemy = PLAYER2 if self.player == PLAYER1 else PLAYER1

        for r in range(BOARD_SIZE - 1):
            for c in range(BOARD_SIZE - 1):
                print("CHECKING DIAGONAL", r, c)
                # Extract the 2x2 grid
                grid = game.board[r:r+2, c:c+2]

                # Check first diagonal: (0,0) and (1,1)
                if (grid[0][0] == enemy and grid[1][1] == enemy and
                    grid[0][1] == EMPTY and grid[1][0] == EMPTY):
                    # Define the positions to block
                    block_positions = [(r, c+1), (r+1, c)]
                    print("DIAGONAL FOUND")
                    # Evaluate blocking moves
                    for move in block_positions:
                        if not game.is_valid_placement(move[0], move[1]):
                            continue  # Skip invalid moves
                        if self.is_block_effective(game, move, enemy):
                            self.logger.debug(f"Blocking enemy diagonal win at {move}")
                            return move

                # Check second diagonal: (0,1) and (1,0)
                if (grid[0][1] == enemy and grid[1][0] == enemy and
                    grid[0][0] == EMPTY and grid[1][1] == EMPTY):
                    # Define the positions to block
                    block_positions = [(r, c), (r+1, c+1)]
                    print("DIAGONAL FOUND")
                    # Evaluate blocking moves
                    for move in block_positions:
                        if not game.is_valid_placement(move[0], move[1]):
                            continue  # Skip invalid moves
                        if self.is_block_effective(game, move, enemy):
                            self.logger.debug(f"Blocking enemy diagonal win at {move}")
                            return move

        # If no blocking move is found
        return None

    def check_and_block_enemy_linear_win(self, game):
        """
        Scans all horizontal and vertical lines on the board to find if the enemy has two
        consecutive pieces that can lead to an immediate win. If such a threat is detected,
        evaluates both blocking positions by simulating each and chooses the one that
        prevents the enemy from having an immediate winning move in their next turn.

        Returns:
            move (list): [row, col] if a blocking move is found, else None.
        """
        enemy = PLAYER2 if self.player == PLAYER1 else PLAYER1
        blocking_moves = []
        print("LINEAR FOUND")
        # Check all horizontal lines
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE - 1):
                print("CHECKING LIENAR", r, c)
                # Check for two consecutive enemy pieces
                if (game.board[r][c] == enemy and game.board[r][c+1] == enemy):
                    # Identify blocking positions at both ends if within bounds
                    blocking_positions = []
                    # Left end
                    if c - 1 >= 0 and game.board[r][c-1] == EMPTY:
                        blocking_positions.append((r, c-1))
                    # Right end
                    if c + 2 < BOARD_SIZE and game.board[r][c+2] == EMPTY:
                        blocking_positions.append((r, c+2))
                    print("LINEAR FOUND")
                    # Evaluate blocking moves
                    for move in blocking_positions:
                        print("DANGER LINEAR FOUND")
                        if not game.is_valid_placement(move[0], move[1]):
                            continue  # Skip invalid moves
                        if self.is_block_effective(game, move, enemy):
                            self.logger.debug(f"Blocking enemy linear win horizontally at {move}")
                            return move

        # Check all vertical lines
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE - 1):
                # Check for two consecutive enemy pieces
                if (game.board[r][c] == enemy and game.board[r+1][c] == enemy):
                    # Identify blocking positions at both ends if within bounds
                    blocking_positions = []
                    # Top end
                    if r - 1 >= 0 and game.board[r-1][c] == EMPTY:
                        blocking_positions.append((r-1, c))
                    # Bottom end
                    if r + 2 < BOARD_SIZE and game.board[r+2][c] == EMPTY:
                        blocking_positions.append((r+2, c))

                    # Evaluate blocking moves
                    for move in blocking_positions:
                        if not game.is_valid_placement(move[0], move[1]):
                            continue  # Skip invalid moves
                        if self.is_block_effective(game, move, enemy):
                            self.logger.debug(f"Blocking enemy linear win vertically at {move}")
                            return move

        # If no blocking move is found
        return None

    def is_block_effective(self, game, move, enemy):
        """
        Simulates placing a blocking move and checks if the enemy can still win immediately.

        Args:
            game (Game): The current game state.
            move (tuple): The blocking move as (row, col).
            enemy (int): The enemy player's identifier.

        Returns:
            bool: True if the blocking move effectively prevents the enemy from winning immediately, else False.
        """
        
        # Simulate the blocking move
        temp_game = copy.deepcopy(game)
        try:
            temp_game.place_checker(move[0], move[1])
        except ValueError as e:
            self.logger.debug(f"Error placing blocker at {move}: {e}")
            return False  # Invalid move, cannot be effective

        # Check if the enemy can still win immediately after the block
        enemy_wins = self.enemy_can_win_immediately(temp_game, enemy)
        return not enemy_wins

    def enemy_can_win_immediately(self, game, enemy):
        """
        Checks if the enemy can win in their next move.

        Args:
            game (Game): The current game state after blocking.
            enemy (int): The enemy player's identifier.

        Returns:
            bool: True if the enemy can win immediately, False otherwise.
        """
        # Temporarily switch to the enemy's turn
        temp_game = copy.deepcopy(game)
        temp_game.current_player = enemy

        # Get all possible moves for the enemy
        enemy_moves = self.get_possible_moves(temp_game)

        # Check if any of the enemy's moves lead to a win
        for move in enemy_moves:
            temp_move_game = copy.deepcopy(temp_game)
            try:
                if len(move) == 2:
                    temp_move_game.place_checker(move[0], move[1])
                elif len(move) == 4:
                    temp_move_game.move_checker(*move)
                else:
                    continue  # Invalid move format
            except ValueError:
                continue  # Skip invalid moves

            if temp_move_game.check_winner() == enemy:
                return True  # Enemy can win immediately

        return False  # Enemy cannot win immediately


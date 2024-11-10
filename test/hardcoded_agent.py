import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy 

class hardcoded_agent:
    def __init__(self, player=PLAYER1):
        self.player = player
    
    # given the game state, gets all of the possible moves
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves
        
    def get_best_move(self, game):
        """Returns a random valid move."""
    
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
                        print("FOUND A WINNER")
                        return [row, col]

            # Check for a block
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if game.board[row][col] != EMPTY:
                        continue
                    temp = copy.deepcopy(game)
                    try:
                        temp.place_checker(row, col)  # Temporarily place enemy's piece
                        temp.board[row][col] = enemy      # Override with enemy's piece
                        temp.current_player = enemy       # Set current player to enemy for win check
                    except ValueError:
                        continue  # Skip invalid placements
                    if temp.check_winner() == enemy:
                        print("FOUND A BLOCK")
                        return [row, col]           

            return random.choice(possible_moves)

        else:
            return random.choice(possible_moves)    

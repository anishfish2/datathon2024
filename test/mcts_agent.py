import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy 
import math
import time
import logging

# ============================== #
#         Logging Setup          #
# ============================== #

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

# Prevent loggers from propagating to the root logger
player1_logger.propagate = False
player2_logger.propagate = False

# ============================== #
#           Node Class           #
# ============================== #

class Node:
    def __init__(self, game_state, parent=None, move=None, agent=None):
        """
        Initialize a Node in the Beam MCTS tree.

        Args:
            game_state (Game): The current game state.
            parent (Node, optional): The parent node. Defaults to None.
            move (tuple, optional): The move that led to this node. Defaults to None.
            agent (MCTSAgent, optional): Reference to the agent to access get_possible_moves. Defaults to None.
        """
        self.game_state = game_state  # Current game state
        self.parent = parent          # Parent node
        self.move = move              # Move that led to this state
        self.children = []            # Child nodes
        self.wins = 0                 # Number of wins from simulations
        self.visits = 0               # Number of times node was visited
        self.agent = agent            # Reference to the agent
        self.untried_moves = self.agent.get_possible_moves(self.game_state) if self.agent else []
        self.player = self.game_state.current_player  # Player who made the move

    def expand(self):
        """Expand the node by creating one child for an untried move."""
        if not self.untried_moves:
            return None  # No moves to expand

        # Select one move to expand (random selection)
        move = self.untried_moves.pop()

        # Apply move to create a new game state
        next_state = copy.deepcopy(self.game_state)
        try:
            if len(move) == 2:
                r, c = move
                if not next_state.is_valid_placement(r, c):
                    self.agent.logger.debug(f"Invalid placement move during expansion: {move}")
                    return self.expand()  # Try expanding another move
                next_state.place_checker(r, c)
            elif len(move) == 4:
                r0, c0, r1, c1 = move
                if not next_state.is_valid_move(*move):
                    self.agent.logger.debug(f"Invalid movement move during expansion: {move}")
                    return self.expand()  # Try expanding another move
                next_state.move_checker(r0, c0, r1, c1)
            else:
                self.agent.logger.debug(f"Unknown move format during expansion: {move}")
                return self.expand()  # Try expanding another move
        except ValueError as e:
            self.agent.logger.debug(f"Error applying move {move} during expansion: {e}")
            return self.expand()  # Try expanding another move

        next_state.turn_count += 1

        # Check for a win
        winner = next_state.check_winner()
        if winner != EMPTY:
            next_state.current_player = EMPTY  # Terminal state
        else:
            next_state.current_player *= -1    # Switch player

        # Create child node
        child_node = Node(next_state, parent=self, move=move, agent=self.agent)
        self.children.append(child_node)
        self.agent.logger.debug(f"Expanded move {move} resulting in new node with winner: {winner}")

        return child_node  # Return the newly created child

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.untried_moves) == 0

    def best_child(self, c_param=math.sqrt(2)):
        """Select the child with the highest UCT value."""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                uct_value = float('inf')  # Prioritize unvisited nodes
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
                uct_value = exploitation + exploration
            choices_weights.append(uct_value)

        if not choices_weights:
            return None  # No children to select from

        # Select the child with the highest UCT value
        best_index = choices_weights.index(max(choices_weights))
        return self.children[best_index]

    def most_visited_child(self):
        """Select the child with the highest visit count."""
        if not self.children:
            return None
        visits = [child.visits for child in self.children]
        best_index = visits.index(max(visits))
        return self.children[best_index]

# ============================== #
#         MCTS Agent Class       #
# ============================== #

class MCTSAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.time_limit = 2  # Time limit per move in seconds
        self.beam_width = 16  # Beam width for Beam MCTS
        self.max_simulation_depth = 0

        # Assign the appropriate logger based on the player
        if self.player == PLAYER1:
            self.logger = player1_logger
        else:
            self.logger = player2_logger

    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.is_valid_placement(r, c):
                        moves.append((r, c))
        else:
            # Movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.is_valid_move(r0, c0, r1, c1):
                                    moves.append((r0, c0, r1, c1))
        self.logger.debug(f"Possible moves generated: {len(moves)} moves")
        return moves

    def validate_move(self, game, move, possible_moves):
        """
        Validates the move. If invalid, attempts to pass or selects a random valid move.

        Args:
            game (Game): The current game state.
            move (list or tuple): The move to validate.
            possible_moves (list): List of all possible valid moves.

        Returns:
            list or tuple: A valid move.
        """
        if self.is_move_valid(game, move):
            return move
        else:
            self.logger.warning(f"Invalid move detected: {move}")
            # Attempt to pass if the game supports it
            if hasattr(game, 'can_pass') and game.can_pass():
                self.logger.info("Attempting to pass the turn.")
                return ['pass']
            else:
                # Select a random valid move
                if possible_moves:
                    random_move = random.choice(possible_moves)
                    self.logger.info(f"Selecting a random valid move: {random_move}")
                    return random_move
                else:
                    self.logger.error("No possible moves available to select.")
                    return None  # No moves available

    def is_move_valid(self, game, move):
        """
        Checks if the move is valid in the current game state.

        Args:
            game (Game): The current game state.
            move (list or tuple): The move to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            if move == ['pass']:
                return hasattr(game, 'can_pass') and game.can_pass()
            if len(move) == 2:
                return game.is_valid_placement(move[0], move[1])
            elif len(move) == 4:
                return game.is_valid_move(*move)
            else:
                return False
        except:
            return False

    def get_best_move(self, game):
        """
        Determines the best move using Beam MCTS, incorporating blocking logic.

        Args:
            game (Game): The current game state.

        Returns:
            list or tuple: The best move to make.
        """
        self.logger.debug("Starting to compute the best move.")
        self.simulation_count = 0
        self.max_simulation_depth = 0
        possible_moves = self.get_possible_moves(game)

        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        enemy = PLAYER2 if self.player == PLAYER1 else PLAYER1

        # ============================== #
        #        Immediate Win Check     #
        # ============================== #

        if current_pieces < NUM_PIECES:
            self.logger.debug("Player is still placing pieces.")
            # Check for a winning move by placing a piece
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
                        self.logger.info(f"FOUND A WINNER by placing at ({row}, {col})")
                        validated_move = self.validate_move(game, [row, col], possible_moves)
                        return validated_move

        # ============================== #
        #      Blocking Enemy Threats    #
        # ============================== #

        # Generalized Threat Checks with Wrap-Around
        blocking_move = self.check_and_block_enemy_threats(game, threat_length=2)
        if blocking_move:
            self.logger.debug(f"Blocking enemy threat by placing at {blocking_move}")
            self.logger.info(f"Blocking enemy threat by placing at {blocking_move}")
            validated_move = self.validate_move(game, blocking_move, possible_moves)
            return validated_move

        # Specific Block: Check if enemy can win immediately and block
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game.board[row][col] != EMPTY:
                    continue
                temp = copy.deepcopy(game)
                try:
                    temp.current_player = enemy       # Set current player to enemy for win check
                    temp.place_checker(row, col)      # Temporarily place enemy's piece
                except ValueError:
                    continue  # Skip invalid placements
                if temp.check_winner() == enemy:
                    self.logger.info(f"FOUND A BLOCK by placing at ({row}, {col})")
                    validated_move = self.validate_move(game, [row, col], possible_moves)
                    return validated_move  

        # ============================== #
        #            Beam MCTS            #
        # ============================== #

        root = Node(copy.deepcopy(game), agent=self)
        beam = [root]

        end_time = time.time() + self.time_limit

        while time.time() < end_time:
            new_beam = []
            for node in beam:
                if not node.is_fully_expanded():
                    # Expand one child
                    child = node.expand()
                    if child:
                        new_beam.append(child)
                else:
                    # Select the best child based on UCT
                    best = node.best_child()
                    if best:
                        new_beam.append(best)

            if not new_beam:
                self.logger.debug("No new nodes to explore in the beam.")
                break  # No nodes to explore

            # Sort the new beam based on visit counts or another heuristic
            new_beam.sort(key=lambda n: n.visits, reverse=True)

            # Keep only the top `beam_width` nodes
            beam = new_beam[:self.beam_width]
            self.logger.debug(f"Beam updated with {len(beam)} nodes.")

            # Perform simulations for each node in the new beam
            for node in beam:
                if node.game_state.current_player == EMPTY:
                    # Handle terminal node: backpropagate the known outcome
                    winner = node.game_state.check_winner()
                    self.backpropagate(node, winner)
                    self.logger.info(f"AFTER TERMINAL BACKPROPAGATE: Visits={node.visits}, Wins={node.wins}")
                    continue  # Skip simulation since the game is over

                winner = self.simulate(node.game_state)
                self.backpropagate(node, winner)
                self.logger.info(f"AFTER MCTS SIM: Visits={node.visits}, Wins={node.wins}")

        # ============================== #
        #       Selecting the Best Move  #
        # ============================== #

        # After Beam MCTS, select the move from the beam with the highest visits
        best_move = None
        max_visits = -1
        for node in beam:
            self.logger.info(f"NODE VISITS: {node.visits}")
            if node.visits > max_visits:
                max_visits = node.visits
                best_move = node.move

        if best_move:
            self.logger.debug(f"Best move selected: {best_move} with {max_visits} visits.")
            # Validate the best move before returning
            validated_move = self.validate_move(game, best_move, possible_moves)
            if validated_move:
                # **Safety Check: Ensure the chosen move doesn't allow the enemy to win immediately**
                if self.is_move_safe(game, validated_move):
                    self.logger.info(f"CHOSE BEST MOVE: {validated_move} with visits: {max_visits}")
                    return validated_move
                else:
                    self.logger.warning(f"Best move {validated_move} is unsafe. Searching for a safe move.")
                    safe_move = self.find_safe_move(game, possible_moves)
                    if safe_move:
                        self.logger.info(f"Selected safe move: {safe_move}")
                        return safe_move
                    else:
                        self.logger.info("No safe move found. Selecting a random move.")
                        random_move = random.choice(possible_moves) if possible_moves else None
                        return self.validate_move(game, random_move, possible_moves)
            else:
                self.logger.error("Validated move is None. Selecting a random move.")
                random_move = random.choice(possible_moves) if possible_moves else None
                return self.validate_move(game, random_move, possible_moves)

        # If no best_move found, select a safe random move
        self.logger.info("No best move found. Attempting to find a safe move.")
        safe_move = self.find_safe_move(game, possible_moves)
        if safe_move:
            self.logger.info(f"Selected safe move: {safe_move}")
            return self.validate_move(game, safe_move, possible_moves)
        else:
            self.logger.info("No safe move found. Selecting a random move.")
            if possible_moves:
                random_move = random.choice(possible_moves)
                return self.validate_move(game, random_move, possible_moves)
            else:
                self.logger.error("No possible moves available to return.")
                return None

    def find_safe_move(self, game, possible_moves):
        """
        Attempts to find a safe move from the list of possible moves.

        Args:
            game (Game): The current game state.
            possible_moves (list): List of possible moves.

        Returns:
            list: A safe move as [row, col] or [r0, c0, r1, c1], else None.
        """
        self.logger.debug("Attempting to find a safe move.")
        safe_moves = []
        for move in possible_moves:
            if self.is_move_safe(game, move):
                safe_moves.append(move)

        if safe_moves:
            selected_move = random.choice(safe_moves)
            self.logger.info(f"Selected safe move: {selected_move}")
            return selected_move
        else:
            self.logger.warning("No safe moves available.")
            return None

    def is_move_safe(self, game, move):
        """
        Checks if making the given move will not allow the enemy to win immediately.

        Args:
            game (Game): The current game state.
            move (tuple): The move to be checked.

        Returns:
            bool: True if the move is safe, False otherwise.
        """
        self.logger.debug(f"Checking if move {move} is safe.")
        temp_game = copy.deepcopy(game)
        try:
            # Apply the move
            if move == ['pass']:
                if hasattr(temp_game, 'pass_turn'):
                    temp_game.pass_turn()
                else:
                    self.logger.debug("Game does not support passing. Move is unsafe.")
                    return False
            elif len(move) == 2:
                temp_game.place_checker(move[0], move[1])
            elif len(move) == 4:
                temp_game.move_checker(*move)
            else:
                self.logger.debug(f"Unknown move format during safety check: {move}")
                return False
            temp_game.turn_count += 1

            # Check for a win
            winner = temp_game.check_winner()
            if winner == self.player:
                self.logger.debug("Move results in a win. It is safe.")
                return True  # Winning move is safe
            elif winner == PLAYER1 or winner == PLAYER2:
                self.logger.debug("Move results in a loss. It is unsafe.")
                return False  # Move leads to a loss
            else:
                # Switch to enemy's turn
                temp_game.current_player = PLAYER2 if self.player == PLAYER1 else PLAYER1

                # Check if enemy can win in their next move
                enemy_wins = self.enemy_can_win_immediately(temp_game, self.player)
                is_safe = not enemy_wins
                self.logger.debug(f"Move safety after enemy's response: {is_safe}")
                return is_safe
        except ValueError as e:
            self.logger.debug(f"Error applying move {move} during safety check: {e}")
            return False  # Invalid move is considered unsafe

    def backpropagate(self, node, winner):
        """Update the node and its ancestors with the simulation result."""
        self.logger.debug(f"Backpropagating result: Winner={winner}")
        while node is not None:
            node.visits += 1
            if winner == self.player:
                node.wins += 1
            elif winner == -self.player:
                node.wins -= 1
            # Optionally, handle draws or neutral outcomes here
            node = node.parent

    def simulate(self, game):
        """Simulates a random play-out from the current game state."""
        self.simulation_count += 1  # Increment simulation count
        current_depth = 0  # Initialize depth for this simulation
        self.logger.debug("Starting simulation.")
        while True:
            winner = game.check_winner()
            if winner != EMPTY:
                if current_depth > self.max_simulation_depth:
                    self.max_simulation_depth = current_depth
                self.logger.debug(f"Simulation ended with winner: {winner}")
                return winner

            possible_moves = self.get_possible_moves(game)
            if not possible_moves:
                if current_depth > self.max_simulation_depth:
                    self.max_simulation_depth = current_depth
                self.logger.debug("Simulation ended in a draw.")
                return EMPTY  # Draw

            move = random.choice(possible_moves)
            self.apply_move(game, move)
            current_depth += 1

            # **Invoke Blocking Logic During Simulation**
            # This ensures that after every simulated move, we attempt to block any new threats
            blocking_move = self.check_and_block_enemy_threats(game, threat_length=2)
            if blocking_move:
                self.logger.debug(f"Simulation: Blocking enemy threat by placing at {blocking_move}")
                self.apply_move(game, blocking_move)

    def apply_move(self, game, move):
        """Applies a move to the game state."""
        try:
            if move == ['pass']:
                if hasattr(game, 'pass_turn'):
                    game.pass_turn()
                else:
                    raise ValueError("Game does not support passing.")
            elif len(move) == 2:
                r, c = move
                game.place_checker(r, c)
            elif len(move) == 4:
                r0, c0, r1, c1 = move
                game.move_checker(r0, c0, r1, c1)
            else:
                raise ValueError(f"Unknown move format: {move}")
            game.turn_count += 1
            # Check for a win
            winner = game.check_winner()
            if winner != EMPTY:
                game.current_player = EMPTY  # No further moves
            else:
                game.current_player *= -1    # Switch player
            self.logger.debug(f"Applied move {move}. Current player is now {game.current_player}.")
        except ValueError as e:
            self.logger.debug(f"Error applying move {move}: {e}")
            raise

    # ============================== #
    #        Blocking Logic          #
    # ============================== #

    def check_and_block_enemy_threats(self, game, threat_length=2):
        """
        Generalized method to check and block enemy threats in all directions,
        including threats that wrap around the grid edges.

        Args:
            game (Game): The current game state.
            threat_length (int): Number of consecutive enemy pieces that constitute a threat.

        Returns:
            move (list): [row, col] if a blocking move is found, else None.
        """
        self.logger.debug("Checking and attempting to block enemy threats.")
        enemy = PLAYER2 if self.player == PLAYER1 else PLAYER1
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1)   # Diagonal down-left
        ]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in directions:
                    threat_blocking_positions = self.detect_threat(game, r, c, dr, dc, enemy, threat_length)
                    if threat_blocking_positions:
                        blocking_move = self.select_blocking_move(game, threat_blocking_positions, enemy)
                        if blocking_move:
                            self.logger.debug(f"Blocking enemy threat by placing at {blocking_move} in direction ({dr}, {dc})")
                            return blocking_move
        return None

    def detect_threat(self, game, start_r, start_c, dr, dc, enemy, threat_length):
        """
        Detects if there's a threat starting from (start_r, start_c) in the direction (dr, dc),
        considering wrap-around if necessary.

        Args:
            game (Game): The current game state.
            start_r (int): Starting row.
            start_c (int): Starting column.
            dr (int): Row direction increment.
            dc (int): Column direction increment.
            enemy (int): Enemy player identifier.
            threat_length (int): Number of consecutive enemy pieces that constitute a threat.

        Returns:
            list: List of blocking positions that can prevent the threat, else None.
        """
        threat_positions = []
        for i in range(threat_length):
            r = (start_r + i * dr) % BOARD_SIZE
            c = (start_c + i * dc) % BOARD_SIZE
            if game.board[r][c] == enemy:
                threat_positions.append((r, c))
            else:
                break

        if len(threat_positions) == threat_length:
            # Check for possible blocking positions before and after the threat
            before_r = (start_r - dr) % BOARD_SIZE
            before_c = (start_c - dc) % BOARD_SIZE
            after_r = (start_r + threat_length * dr) % BOARD_SIZE
            after_c = (start_c + threat_length * dc) % BOARD_SIZE
            blocking_positions = []

            if game.board[before_r][before_c] == EMPTY:
                blocking_positions.append((before_r, before_c))

            if game.board[after_r][after_c] == EMPTY:
                blocking_positions.append((after_r, after_c))

            if blocking_positions:
                return blocking_positions
        return None

    def select_blocking_move(self, game, threat_blocking_positions, enemy):
        """
        Selects a blocking move from the available blocking positions,
        ensuring that the move is valid and effective.

        Args:
            game (Game): The current game state.
            threat_blocking_positions (list): List of blocking positions.
            enemy (int): Enemy player identifier.

        Returns:
            list: [row, col] of the blocking move, else None.
        """
        self.logger.debug("Selecting a blocking move from potential blocking positions.")
        for move in threat_blocking_positions:
            r, c = move
            if game.is_valid_placement(r, c):
                if self.is_block_effective(game, move, enemy):
                    return list(move)
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
        self.logger.debug(f"Checking if blocking move {move} is effective.")
        # Simulate the blocking move
        temp_game = copy.deepcopy(game)
        try:
            temp_game.place_checker(move[0], move[1])
        except ValueError as e:
            self.logger.debug(f"Error placing blocker at {move}: {e}")
            return False  # Invalid move, cannot be effective

        # Check if the enemy can still win immediately after the block
        enemy_wins = self.enemy_can_win_immediately(temp_game, enemy)
        is_effective = not enemy_wins
        self.logger.debug(f"Blocking move effectiveness: {is_effective}")
        return is_effective

    def enemy_can_win_immediately(self, game, enemy):
        """
        Checks if the enemy can win in their next move.

        Args:
            game (Game): The current game state after blocking.
            enemy (int): The enemy player's identifier.

        Returns:
            bool: True if the enemy can win immediately, False otherwise.
        """
        self.logger.debug("Checking if the enemy can win immediately after the block.")
        # Temporarily switch to the enemy's turn
        temp_game = copy.deepcopy(game)
        temp_game.current_player = enemy

        # Get all possible moves for the enemy
        enemy_moves = self.get_possible_moves(temp_game)

        # Check if any of the enemy's moves lead to a win
        for move in enemy_moves:
            temp_move_game = copy.deepcopy(temp_game)
            try:
                if move == ['pass']:
                    if hasattr(temp_move_game, 'pass_turn'):
                        temp_move_game.pass_turn()
                    else:
                        continue  # Skip if pass is not supported
                elif len(move) == 2:
                    temp_move_game.place_checker(move[0], move[1])
                elif len(move) == 4:
                    temp_move_game.move_checker(*move)
                else:
                    continue  # Invalid move format
            except ValueError:
                continue  # Skip invalid moves

            if temp_move_game.check_winner() == enemy:
                self.logger.debug(f"Enemy can win immediately with move: {move}")
                return True  # Enemy can win immediately

        self.logger.debug("Enemy cannot win immediately after the block.")
        return False  # Enemy cannot win immediately

    # ============================== #
    #       Additional Methods       #
    # ============================== #

    def find_safe_move(self, game, possible_moves):
        """
        Attempts to find a safe move from the list of possible moves.

        Args:
            game (Game): The current game state.
            possible_moves (list): List of possible moves.

        Returns:
            list: A safe move as [row, col] or [r0, c0, r1, c1], else None.
        """
        self.logger.debug("Attempting to find a safe move.")
        safe_moves = []
        for move in possible_moves:
            if self.is_move_safe(game, move):
                safe_moves.append(move)

        if safe_moves:
            selected_move = random.choice(safe_moves)
            self.logger.info(f"Selected safe move: {selected_move}")
            return selected_move
        else:
            self.logger.warning("No safe moves available.")
            return None

    # ============================== #
    #         Safety Checks          #
    # ============================== #

    def is_move_safe(self, game, move):
        """
        Checks if making the given move will not allow the enemy to win immediately.

        Args:
            game (Game): The current game state.
            move (tuple): The move to be checked.

        Returns:
            bool: True if the move is safe, False otherwise.
        """
        self.logger.debug(f"Checking if move {move} is safe.")
        temp_game = copy.deepcopy(game)
        try:
            # Apply the move
            if move == ['pass']:
                if hasattr(temp_game, 'pass_turn'):
                    temp_game.pass_turn()
                else:
                    self.logger.debug("Game does not support passing. Move is unsafe.")
                    return False
            elif len(move) == 2:
                temp_game.place_checker(move[0], move[1])
            elif len(move) == 4:
                temp_game.move_checker(*move)
            else:
                self.logger.debug(f"Unknown move format during safety check: {move}")
                return False
            temp_game.turn_count += 1

            # Check for a win
            winner = temp_game.check_winner()
            if winner == self.player:
                self.logger.debug("Move results in a win. It is safe.")
                return True  # Winning move is safe
            elif winner == PLAYER1 or winner == PLAYER2:
                self.logger.debug("Move results in a loss. It is unsafe.")
                return False  # Move leads to a loss
            else:
                # Switch to enemy's turn
                temp_game.current_player = PLAYER2 if self.player == PLAYER1 else PLAYER1

                # Check if enemy can win in their next move
                enemy_wins = self.enemy_can_win_immediately(temp_game, self.player)
                is_safe = not enemy_wins
                self.logger.debug(f"Move safety after enemy's response: {is_safe}")
                return is_safe
        except ValueError as e:
            self.logger.debug(f"Error applying move {move} during safety check: {e}")
            return False  # Invalid move is considered unsafe

    # ============================== #
    #         Simulation Steps       #
    # ============================== #

    def simulate(self, game):
        """Simulates a random play-out from the current game state."""
        self.simulation_count += 1  # Increment simulation count
        current_depth = 0  # Initialize depth for this simulation
        self.logger.debug("Starting simulation.")
        while True:
            winner = game.check_winner()
            if winner != EMPTY:
                if current_depth > self.max_simulation_depth:
                    self.max_simulation_depth = current_depth
                self.logger.debug(f"Simulation ended with winner: {winner}")
                return winner

            possible_moves = self.get_possible_moves(game)
            if not possible_moves:
                if current_depth > self.max_simulation_depth:
                    self.max_simulation_depth = current_depth
                self.logger.debug("Simulation ended in a draw.")
                return EMPTY  # Draw

            move = random.choice(possible_moves)
            self.apply_move(game, move)
            current_depth += 1

            # **Invoke Blocking Logic During Simulation**
            # This ensures that after every simulated move, we attempt to block any new threats
            blocking_move = self.check_and_block_enemy_threats(game, threat_length=2)
            if blocking_move:
                self.logger.debug(f"Simulation: Blocking enemy threat by placing at {blocking_move}")
                self.apply_move(game, blocking_move)

    # ============================== #
    #       Threat Detection         #
    # ============================== #

    # (Already defined above in Blocking Logic)


import pygame
import sys
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE
from hardcoded_agent import hardcoded_agent

# Constants
CELL_SIZE = 80
MARGIN = 20
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()
pygame.display.set_caption("PushBattle")
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Load images or define piece representations
def draw_board(game):
    screen.fill(GRAY)
    # Draw grid
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)
            piece = game.board[row][col]
            if piece != EMPTY:
                center = rect.center
                if piece == PLAYER1:
                    pygame.draw.circle(screen, WHITE, center, CELL_SIZE//2 - 10)
                elif piece == PLAYER2:
                    pygame.draw.circle(screen, BLACK, center, CELL_SIZE//2 - 10)

def get_cell(pos):
    x, y = pos
    if x < MARGIN or y < MARGIN:
        return None
    col = (x - MARGIN) // CELL_SIZE
    row = (y - MARGIN) // CELL_SIZE
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None

def display_message(message):
    text = font.render(message, True, RED)
    rect = text.get_rect(center=(WINDOW_SIZE//2, MARGIN//2))
    screen.blit(text, rect)

def main():
    game = Game()
    agent = hardcoded_agent(player=PLAYER2)  # AI plays as PLAYER2
    human_player = PLAYER1
    ai_player = PLAYER2
    game_over = False
    winner = EMPTY
    move_mode = 'placement'  # or 'movement'
    selected_piece = None  # For movement
    messages = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if not game_over and game.current_player == human_player:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    cell = get_cell(pos)
                    if cell:
                        row, col = cell
                        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
                        if current_pieces < 8:
                            # Placement mode
                            if game.is_valid_placement(row, col):
                                game.place_checker(row, col)
                                game.turn_count += 1
                                winner = game.check_winner()
                                if winner != EMPTY:
                                    game_over = True
                                else:
                                    game.current_player *= -1
                        else:
                            # Movement mode
                            if selected_piece is None:
                                if game.board[row][col] == human_player:
                                    selected_piece = (row, col)
                                    messages.append(f"Selected piece at ({row}, {col})")
                            else:
                                src_row, src_col = selected_piece
                                if (row, col) == selected_piece:
                                    selected_piece = None
                                    messages.append("Deselected piece")
                                elif game.is_valid_move(src_row, src_col, row, col):
                                    game.move_checker(src_row, src_col, row, col)
                                    game.turn_count += 1
                                    winner = game.check_winner()
                                    if winner != EMPTY:
                                        game_over = True
                                    else:
                                        game.current_player *= -1
                                    selected_piece = None
                                else:
                                    messages.append("Invalid move")
        
        # AI's turn
        if not game_over and game.current_player == ai_player:
            pygame.display.flip()
            pygame.display.update()
            pygame.time.delay(500)  # Small delay for better visualization
            move = agent.get_best_move(game)
            if len(move) == 2:
                r, c = move
                if game.is_valid_placement(r, c):
                    game.place_checker(r, c)
                    messages.append(f"AI placed at ({r}, {c})")
            elif len(move) == 4:
                r0, c0, r1, c1 = move
                if game.is_valid_move(r0, c0, r1, c1):
                    game.move_checker(r0, c0, r1, c1)
                    messages.append(f"AI moved from ({r0}, {c0}) to ({r1}, {c1})")
            game.turn_count += 1
            winner = game.check_winner()
            if winner != EMPTY:
                game_over = True
            else:
                game.current_player *= -1

        # Drawing
        draw_board(game)

        # Highlight selected piece
        if selected_piece:
            row, col = selected_piece
            rect = pygame.Rect(MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect, 3)

        # Display messages
        if messages:
            for idx, msg in enumerate(messages[-5:]):  # Show last 5 messages
                text = font.render(msg, True, BLUE)
                screen.blit(text, (MARGIN, WINDOW_SIZE - MARGIN + 5 + idx * 20))

        # Display winner
        if game_over:
            if winner == PLAYER1:
                display_message("You Win!")
            elif winner == PLAYER2:
                display_message("AI Wins!")
            else:
                display_message("It's a Draw!")

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

import sys
from typing import List, Tuple, Union, Dict

# Typing aliases for clarity
Board = List[int]
Move = Tuple[int, int]  # Tuple of (pile index, number of sticks to remove)
Score = Union[int, float] # Union type for score (int or float)


class Nim(object):
    def __init__(self, board):
        self.board = board
        # Since this has a lot of overlapping states, we can use a cache to store the results of previously computed states
        self.cache: Dict[Tuple[Tuple[int, ...], int, bool], Tuple[Score, Move]] = {}

    # Implement any additional functions needed here

    # Useful constants
    INF: Score = float('inf')  # Infinity value for alpha-beta pruning
    WIN: Score = sys.maxsize  # Winning value
    LOSS: Score = -WIN  # Losing value
    MAX_DEPTH: int = 5  # Maximum depth for minimax search (used for efficient searching)

    def is_game_over(self, board: Board) -> bool:
        return sum(board) == 0 or sum(board) == 1

    def make_move(self, board: Board, move: Move) -> Board:
        new_board = board.copy()
        pile_idx, remove_sticks = move
        new_board[pile_idx] -= remove_sticks
        return new_board

    # The nim sum helps determine winning and losing positions 
    # This can be used as a heuristic and limiting the depth of the search tree so that the entire tree does not need to be searched
    # (credits to https://en.wikipedia.org/wiki/Nim#Proof_of_the_winning_formula)
    def nim_sum(self, board: Board) -> int:
        nim_sum = 0
        for pile in board:
            nim_sum ^= pile
        return nim_sum

    def generate_valid_moves(self, board: Board) -> List[Move]:
        valid_moves: List[Move] = []
        for pile_idx, sticks in enumerate(board):
            for remove_sticks in range(1, sticks + 1):
                valid_moves.append((pile_idx, remove_sticks))
        return valid_moves
    
    def evaluate(self, board: Board) -> Score:
        # If terminal state, return WIN or LOSS value
        if self.is_game_over(board):
            return self.WIN if sum(board) == 0 else self.LOSS
        
        # If not terminal, return the nim sum
        return self.nim_sum(board)

    def minimax(self, board: Board, depth: int, maximizing_player: bool, alpha: Score, beta: Score) -> Tuple[Score, Move]:
        """
        Minimax with alpha-beta pruning algorithm

        Parameters:
        - board: 1d matrix where each entry represents pile and value in the entry represents number of sticks
        - depth: depth
        - maximizing_player: boolean which is equal to True when the player tries to maximize the score
        - alpha: alpha variable for pruning
        - beta: beta variable for pruning 
        Returns:
        - Best value (as a Score)
        - Everything needed to identify next move (returns a tuple of pile index and number of sticks to remove)

        """
        
        # Best value for the maximising player
        def max_value(board: Board, depth: int, alpha: Score, beta: Score) -> Tuple[Score, Move]:
            # Convert board to tuple for caching
            board_tuple = tuple(board)
            cache_key = (board_tuple, depth, True)
            
            # Check if this state is already in cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if self.is_game_over(board) or depth >= self.MAX_DEPTH:
                result = (self.evaluate(board), (-1, -1))
                self.cache[cache_key] = result
                return result
            
            v_curr: Score = -self.INF
            best_move: Move = (-1, -1)

            for next_move in self.generate_valid_moves(board):
                next_state: Board = self.make_move(board, next_move)
                min_val, _ = min_value(next_state, depth + 1, alpha, beta)
                if min_val > v_curr:
                    v_curr = min_val
                    best_move = next_move
                if v_curr >= beta:
                    result = (v_curr, best_move)
                    self.cache[cache_key] = result
                    return result
                alpha = max(alpha, v_curr)
                
            result = (v_curr, best_move)
            self.cache[cache_key] = result
            return result
        
        # Best value for the minimising player
        def min_value(board: Board, depth: int, alpha: Score, beta: Score) -> Tuple[Score, Move]:
            # Convert board to tuple for caching
            board_tuple = tuple(board)
            cache_key = (board_tuple, depth, False)
            
            # Check if this state is already in cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if self.is_game_over(board) or depth >= self.MAX_DEPTH:
                result = (-self.evaluate(board), (-1, -1))
                self.cache[cache_key] = result
                return result
            
            v_curr: Score = self.INF
            best_move: Move = (-1, -1)

            for next_move in self.generate_valid_moves(board):
                next_state: Board = self.make_move(board, next_move)
                max_val, _ = max_value(next_state, depth + 1, alpha, beta)
                if max_val < v_curr:
                    v_curr = max_val
                    best_move = next_move
                if v_curr <= alpha:
                    result = (v_curr, best_move)
                    self.cache[cache_key] = result
                    return result
                beta = min(beta, v_curr)
                
            result = (v_curr, best_move)
            self.cache[cache_key] = result
            return result
        
        if maximizing_player:
            return max_value(board, depth, alpha, beta)
        else:
            return min_value(board, depth, alpha, beta)

if __name__ == "__main__":
    """
    Main game loop
    Firtsly, allow player to choose how many piles will be in the game and number of sticks in each pile

    Implement the game loop
    """

    print("Starting Nim!")

    # initializing size of the game board
    ele = int(input("Input the number of piles "))

    # Check if the number of piles is valid
    if ele <= 0:
        print("Invalid number of piles. Exiting.")
        exit(1)

    piles = []

    print("Input the number of sticks in each pile (separate with ENTER)")
    for _ in range(ele):
        piles.append(int(input()))

    game = Nim(piles)
    print("Enter the pile to remove from (starting from 0), then space followed by enter the number of sticks to remove")
    print("The person who removes the last stick loses!")
    print("Example: to remove from 2nd pile 3 sticks , enter 2 3")
    
    while True:
        print("Pile state %s" % (game.board))
        
        # Human player's turn
        try:
            player_input = input("Your move: ")
            pile_idx, remove_sticks = map(int, player_input.split())
            
            # Validate move
            if pile_idx < 0 or pile_idx >= len(game.board):
                print("Invalid pile index. Try again.")
                continue
            
            if remove_sticks <= 0 or remove_sticks > game.board[pile_idx]:
                print("Invalid number of sticks to remove. Try again.")
                continue
            
            # Apply human's move
            game.board[pile_idx] -= remove_sticks
            
            # Check if game is over after human's move
            if game.is_game_over(game.board):
                if game.evaluate(game.board) == game.WIN:
                    print("Game over! You lose (removed the last stick).")
                else:
                    print("Game over! You win.")
                break
            
            print("Pile state after your move: %s" % (game.board))
            
            # AI's turn
            print("AI is thinking...")
            score, ai_move = game.minimax(game.board, 0, False, -game.INF, game.INF)
            
            pile_idx, remove_sticks = ai_move
            print(f"AI removes {remove_sticks} sticks from pile {pile_idx}")
            
            # Apply AI's move
            game.board[pile_idx] -= remove_sticks
            
            # Check if game is over after AI's move
            if game.is_game_over(game.board):
                if score == game.WIN:
                    print("Game over! AI loses (removed the last stick).")
                else:
                    print("Game over! AI wins.")
                break
            
        except ValueError:
            print("Invalid input format. Please enter two integers separated by a space.")
            continue
        except IndexError:
            print("Invalid input. Try again.")
            continue
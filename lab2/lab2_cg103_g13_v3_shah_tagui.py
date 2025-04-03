import sys

def minimax(board, depth, maximizing_player, alpha, beta):
    if sum(board) == 0:
        return (-1, -1) if maximizing_player else (-1, 1)
    
    if maximizing_player:
        max_eval = -sys.maxsize
        best_move = (-1, -1)
        for i in range(len(board)):
            if board[i] == 0:
                continue
            for j in range(1, board[i] + 1):
                new_board = board[:]
                new_board[i] -= j
                _, eval = minimax(new_board, depth + 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (i, j)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return best_move, max_eval
    else:
        min_eval = sys.maxsize
        best_move = (-1, -1)
        for i in range(len(board)):
            if board[i] == 0:
                continue
            for j in range(1, board[i] + 1):
                new_board = board[:]
                new_board[i] -= j
                _, eval = minimax(new_board, depth + 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = (i, j)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return best_move, min_eval

class Nim:
    def __init__(self, board):
        self.board = board[:]
    
    def make_move(self, pile, sticks):
        if 0 <= pile < len(self.board) and 1 <= sticks <= self.board[pile]:
            self.board[pile] -= sticks
            return True
        return False
    
    def is_game_over(self):
        return all(sticks == 0 for sticks in self.board)

if __name__ == "__main__":
    print("Starting Nim!")
    num_piles = int(input("Enter number of piles: "))
    piles = []
    for i in range(num_piles):
        while True:
            try:
                sticks = int(input(f"Enter sticks in pile {i}: "))
                if sticks < 0:
                    raise ValueError
                piles.append(sticks)
                break
            except ValueError:
                print("Invalid input. Please enter a non-negative integer.")
    
    game = Nim(piles)
    
    while not game.is_game_over():
        print("Current board state:", game.board)
        while True:
            try:
                move = input("Enter your move (pile index and number of sticks): ")
                pile, sticks = map(int, move.split())
                if game.make_move(pile, sticks):
                    break
                else:
                    print("Invalid move. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid indices and values.")
        
        if game.is_game_over():
            print("You lost! AI wins.")
            break
        
        print("AI is thinking...")
        ai_move, _ = minimax(game.board, 0, True, -sys.maxsize, sys.maxsize)
        if ai_move == (-1, -1):
            print("AI has no valid moves left. You win!")
            break
        print(f"AI removes {ai_move[1]} sticks from pile {ai_move[0]}")
        game.make_move(ai_move[0], ai_move[1])
        
        if game.is_game_over():
            print("AI lost! You win.")

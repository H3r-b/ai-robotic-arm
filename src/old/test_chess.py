import chess
import chess.engine
import os

# Get correct path
base_path = os.path.dirname(os.path.dirname(__file__))
stockfish_path = os.path.join(base_path, "stockfish", "stockfish-windows-x86-64-avx2.exe")

print("Stockfish path:", stockfish_path)

# Load engine
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

board = chess.Board()

print("Initial Board:")
print(board)

# Get best move
result = engine.play(board, chess.engine.Limit(time=0.5))

print("Best move:", result.move)

engine.quit()
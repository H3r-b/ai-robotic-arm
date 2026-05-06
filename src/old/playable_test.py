import cv2
import numpy as np
import chess
import chess.engine
import os

# ======================
# PATH SETUP
# ======================
base_path = os.path.dirname(os.path.dirname(__file__))

stockfish_path = os.path.join(
    base_path,
    "stockfish",
    "stockfish-windows-x86-64-avx2.exe"
)

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
board = chess.Board()

# ======================
# BOARD SETTINGS
# ======================
BOARD_SIZE = 800
PADDING = 60
INNER_SIZE = BOARD_SIZE - 2 * PADDING
sq = INNER_SIZE // 8

# ======================
# LOAD PIECES
# ======================
def load_pieces(size):
    pieces = {}
    pieces_path = os.path.join(base_path, "pieces")

    for name in ["wp","wr","wn","wb","wq","wk",
                 "bp","br","bn","bb","bq","bk"]:
        img = cv2.imread(os.path.join(pieces_path, f"{name}.png"), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise Exception(f"Missing: {name}.png")

        img = cv2.resize(img, (size, size))
        pieces[name] = img

    return pieces

piece_imgs = load_pieces(sq)

# ======================
# OVERLAY FUNCTION
# ======================
def overlay_png(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * bg[y:y+h, x:x+w, c]
        )

# ======================
# FORMAT MOVE
# ======================
def format_move(move):
    return f"{move.uci()[:2]} → {move.uci()[2:]}"

# ======================
# GET LEGAL MOVES
# ======================
def get_legal_moves(square):
    return [move.to_square for move in board.legal_moves if move.from_square == square]

# ======================
# DRAW BOARD (UPDATED UI)
# ======================
def draw_board():
    gui = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    # Black border background
    gui[:] = (0, 0, 0)

    # Squares
    for row in range(8):
        for col in range(8):
            color = (181,217,240) if (row+col)%2==0 else (89,135,180)

            x1 = PADDING + col * sq
            y1 = PADDING + row * sq
            x2 = x1 + sq
            y2 = y1 + sq

            cv2.rectangle(gui, (x1,y1), (x2,y2), color, -1)

    # ======================
    # COORDINATES
    # ======================
    letters = "ABCDEFGH"

    for i in range(8):
        # Bottom
        cv2.putText(gui, letters[i],
                    (PADDING + i*sq + sq//2 - 10, BOARD_SIZE - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Top
        cv2.putText(gui, letters[i],
                    (PADDING + i*sq + sq//2 - 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Left
        cv2.putText(gui, str(8-i),
                    (15, PADDING + i*sq + sq//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Right
        cv2.putText(gui, str(8-i),
                    (BOARD_SIZE - 40, PADDING + i*sq + sq//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ======================
    # PIECES
    # ======================
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8

            name = piece.symbol()
            name = ("w" if name.isupper() else "b") + name.lower()

            x = PADDING + col * sq
            y = PADDING + row * sq

            overlay_png(gui, piece_imgs[name], x, y)

    return gui

# ======================
# STATE
# ======================
selected_square = None
legal_moves = []

# ======================
# MOUSE
# ======================
def mouse_callback(event, x, y, flags, param):
    global selected_square, legal_moves, board

    if event == cv2.EVENT_LBUTTONDOWN:
        if PADDING < x < BOARD_SIZE-PADDING and PADDING < y < BOARD_SIZE-PADDING:

            col = (x - PADDING) // sq
            row = (y - PADDING) // sq
            square = chess.square(col, 7 - row)

            if selected_square is None:
                piece = board.piece_at(square)

                if piece and piece.color == chess.WHITE:
                    selected_square = square
                    legal_moves = get_legal_moves(square)

            else:
                move = chess.Move(selected_square, square)

                if move in board.legal_moves:
                    print(f"Player: {format_move(move)}")
                    board.push(move)

                    result = engine.play(board, chess.engine.Limit(time=0.3))
                    ai_move = result.move

                    print(f"AI: {format_move(ai_move)}")

                    board.push(ai_move)

                    print(board)
                    print("-" * 40)

                selected_square = None
                legal_moves = []

# ======================
# MAIN LOOP
# ======================
cv2.namedWindow("Chess AI")
cv2.setMouseCallback("Chess AI", mouse_callback)

while True:
    gui = draw_board()

    # 🟩 Legal moves
    for sqr in legal_moves:
        row = 7 - (sqr // 8)
        col = sqr % 8

        x = PADDING + col * sq
        y = PADDING + row * sq

        cv2.rectangle(gui, (x, y), (x+sq, y+sq), (0,255,0), 3)

    # 🟨 Selected square
    if selected_square is not None:
        row = 7 - (selected_square // 8)
        col = selected_square % 8

        x = PADDING + col * sq
        y = PADDING + row * sq

        cv2.rectangle(gui, (x,y), (x+sq,y+sq), (0,255,255), 3)

    cv2.imshow("Chess AI", gui)

    if cv2.waitKey(1) & 0xFF == 27:
        break

engine.quit()
cv2.destroyAllWindows()
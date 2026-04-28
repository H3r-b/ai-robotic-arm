import cv2
import numpy as np

THRESHOLD_DIFF = 40


# ---- detect if piece exists ----
def is_piece_present(empty_cell, current_cell):
    empty_gray = cv2.cvtColor(empty_cell, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_cell, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(empty_gray, curr_gray)
    _, thresh = cv2.threshold(diff, THRESHOLD_DIFF, 255, cv2.THRESH_BINARY)

    return np.sum(thresh == 255) > 300


# ---- detect color ----
def detect_color(cell):
    avg = np.mean(cell)
    return "white" if avg > 130 else "black"


# ---- detect piece type (simple heuristic) ----
def detect_piece_type(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "unknown"

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / (w + 1e-5)

    # ---- VERY BASIC RULES ----
    if area < 500:
        return "pawn"
    elif aspect_ratio > 1.5:
        return "bishop"
    elif 0.8 < aspect_ratio < 1.2:
        return "rook"
    elif area > 1500:
        return "queen"
    else:
        return "knight"


# ---- main board detection ----
def get_board_state(empty_frame, current_frame):
    board = [["" for _ in range(8)] for _ in range(8)]

    cell_size = 80

    for i in range(8):
        for j in range(8):
            empty_cell = empty_frame[i*cell_size:(i+1)*cell_size,
                                     j*cell_size:(j+1)*cell_size]

            curr_cell = current_frame[i*cell_size:(i+1)*cell_size,
                                      j*cell_size:(j+1)*cell_size]

            # center crop
            h, w, _ = curr_cell.shape
            empty_cell = empty_cell[h//4:3*h//4, w//4:3*w//4]
            curr_cell = curr_cell[h//4:3*h//4, w//4:3*w//4]

            if not is_piece_present(empty_cell, curr_cell):
                board[i][j] = "."
            else:
                color = detect_color(curr_cell)
                piece = detect_piece_type(curr_cell)

                board[i][j] = f"{color}_{piece}"

    return board
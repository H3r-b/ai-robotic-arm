import cv2
from identify_pieces import get_board_state

empty = cv2.imread("empty_board.jpg")
current = cv2.imread("fixed_board.jpg")

board = get_board_state(empty, current)

for row in board:
    print(row)
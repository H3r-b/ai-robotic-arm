import cv2
import numpy as np

CHESSBOARD_SIZE = (7, 7)
BOARD_SIZE = 800
PADDING = 60

# ======================
# LOAD PIECE IMAGES (ONCE)
# ======================
import os

def load_pieces(size):
    pieces = {}

    base_path = os.path.dirname(os.path.dirname(__file__))  # go up from /src
    pieces_path = os.path.join(base_path, "pieces")

    names = ["wp","wr","wn","wb","wq","wk",
             "bp","br","bn","bb","bq","bk"]

    for name in names:
        img_path = os.path.join(pieces_path, f"{name}.png")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise Exception(f"Missing image: {img_path}")

        img = cv2.resize(img, (size, size))
        pieces[name] = img

    return pieces


# ======================
# OVERLAY FUNCTION (FAST)
# ======================
def overlay_png(bg, overlay, x, y):
    h, w = overlay.shape[:2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )
    else:
        bg[y:y+h, x:x+w] = overlay


# ======================
# INIT
# ======================
INNER_SIZE = BOARD_SIZE - 2 * PADDING
sq = INNER_SIZE // 8

piece_imgs = load_pieces(sq)

cap = cv2.VideoCapture(0)

# Default board setup
board = [
    ["br","bn","bb","bq","bk","bb","bn","br"],
    ["bp","bp","bp","bp","bp","bp","bp","bp"],
    ["","","","","","","",""],
    ["","","","","","","",""],
    ["","","","","","","",""],
    ["","","","","","","",""],
    ["wp","wp","wp","wp","wp","wp","wp","wp"],
    ["wr","wn","wb","wq","wk","wb","wn","wr"]
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ======================
    # GUI BOARD
    # ======================
    gui = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    gui[:] = (0, 0, 0)

    # Draw squares
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
    letters = "abcdefgh"

    for i in range(8):
        cv2.putText(gui, letters[i],
                    (PADDING + i*sq + sq//2 - 10, BOARD_SIZE - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(gui, str(8-i),
                    (15, PADDING + i*sq + sq//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ======================
    # DRAW PIECES (PNG)
    # ======================
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != "":
                x = PADDING + col * sq
                y = PADDING + row * sq
                overlay_png(gui, piece_imgs[piece], x, y)

    # ======================
    # DETECTION
    # ======================
    warp = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_corners:
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        corners = corners.reshape(-1, 2)

        tl = corners[0]
        tr = corners[CHESSBOARD_SIZE[0]-1]
        bl = corners[-CHESSBOARD_SIZE[0]]
        br = corners[-1]

        dx = (tr - tl) / (CHESSBOARD_SIZE[0]-1)
        dy = (bl - tl) / (CHESSBOARD_SIZE[1]-1)

        tl_ext = tl - dx - dy
        tr_ext = tr + dx - dy
        bl_ext = bl - dx + dy
        br_ext = br + dx + dy

        pts_src = np.array([tl_ext, tr_ext, br_ext, bl_ext], dtype="float32")

        cv2.polylines(display_frame, [pts_src.astype(int)], True, (0,255,0), 3)

        pts_dst = np.array([
            [0, 0],
            [BOARD_SIZE, 0],
            [BOARD_SIZE, BOARD_SIZE],
            [0, BOARD_SIZE]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warp = cv2.warpPerspective(frame, matrix, (BOARD_SIZE, BOARD_SIZE))

        # OPTIONAL: simple detection overlay
        square_size = BOARD_SIZE // 8

        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size

                square = warp[y1:y2, x1:x2]
                gray_sq = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)

                if np.mean(gray_sq) < 100:
                    cx = PADDING + col * sq + sq // 2
                    cy = PADDING + row * sq + sq // 2
                    cv2.circle(gui, (cx, cy), 15, (0,0,255), -1)

                cv2.rectangle(warp, (x1,y1), (x2,y2), (0,255,0), 1)

    else:
        cv2.putText(gui, "Waiting for board...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    # ======================
    # LAYOUT
    # ======================
    screen_w = 1400
    screen_h = 800

    left_w = int(screen_w * 0.7)
    right_w = screen_w - left_w
    right_h_half = screen_h // 2

    left_big = cv2.resize(gui, (left_w, screen_h))
    right_top = cv2.resize(warp, (right_w, right_h_half))
    right_bottom = cv2.resize(display_frame, (right_w, right_h_half))

    right_stack = np.vstack((right_top, right_bottom))
    final = np.hstack((left_big, right_stack))

    cv2.namedWindow("Chess System", cv2.WINDOW_NORMAL)
    cv2.imshow("Chess System", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
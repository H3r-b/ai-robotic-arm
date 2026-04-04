import cv2
import numpy as np

CHESSBOARD_SIZE = (7, 7)
BOARD_SIZE = 800

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    warp = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    gui = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    if ret_corners:
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        corners = corners.reshape(-1, 2)

        # Inner corners
        tl = corners[0]
        tr = corners[CHESSBOARD_SIZE[0]-1]
        bl = corners[-CHESSBOARD_SIZE[0]]
        br = corners[-1]

        # Direction vectors
        dx = (tr - tl) / (CHESSBOARD_SIZE[0]-1)
        dy = (bl - tl) / (CHESSBOARD_SIZE[1]-1)

        # Expand to full board
        tl_ext = tl - dx - dy
        tr_ext = tr + dx - dy
        bl_ext = bl - dx + dy
        br_ext = br + dx + dy

        pts_src = np.array([tl_ext, tr_ext, br_ext, bl_ext], dtype="float32")

        # Draw green outline
        cv2.polylines(display_frame, [pts_src.astype(int)], True, (0,255,0), 3)

        # Perspective transform
        pts_dst = np.array([
            [0, 0],
            [BOARD_SIZE, 0],
            [BOARD_SIZE, BOARD_SIZE],
            [0, BOARD_SIZE]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warp = cv2.warpPerspective(frame, matrix, (BOARD_SIZE, BOARD_SIZE))

        # ======================
        # GRID SPLIT
        # ======================
        square_size = BOARD_SIZE // 8
        squares = []

        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size

                square = warp[y1:y2, x1:x2]
                squares.append(square)

                # Draw grid
                cv2.rectangle(warp, (x1, y1), (x2, y2), (0,255,0), 1)

        # ======================
        # DIGITAL BOARD (Brown & Beige)
        # ======================
        sq = BOARD_SIZE // 8

        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    color = (170, 200, 220)   # beige (BGR)
                else:
                    color = (40, 70, 120)     # brown (BGR)

                x1 = col * sq
                y1 = row * sq
                x2 = x1 + sq
                y2 = y1 + sq

                cv2.rectangle(gui, (x1, y1), (x2, y2), color, -1)

        # ======================
        # LABELS (A-H, 1-8)
        # ======================
        letters = "ABCDEFGH"

        for i in range(8):
            # Bottom letters
            cv2.putText(gui, letters[i],
                        (i * sq + sq//2 - 10, BOARD_SIZE - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # Left numbers
            cv2.putText(gui, str(8 - i),
                        (5, i * sq + sq//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # ======================
        # SIMPLE PIECE DETECTION
        # ======================
        for i, square in enumerate(squares):
            row = i // 8
            col = i % 8

            gray_sq = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_sq) < 100:
                cx = col * sq + sq // 2
                cy = row * sq + sq // 2
                cv2.circle(gui, (cx, cy), 20, (0,0,255), -1)

    # ======================
    # RESPONSIVE LAYOUT
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

    # Optional fullscreen (uncomment if needed)
    # cv2.setWindowProperty("Chess System",
    #                      cv2.WND_PROP_FULLSCREEN,
    #                      cv2.WINDOW_FULLSCREEN)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
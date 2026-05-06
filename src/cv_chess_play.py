import cv2
import json
import numpy as np
import chess
import chess.engine
import chess.svg
from PIL import Image
import io
import random
import os
import sys
import cairosvg
import time

# =========================
# CONFIG
# =========================
CALIB_JSON = "sqdict.json"

ENGINE_PATH = r"stockfish/stockfish-windows-x86-64-avx2.exe"

MOVE_THRESHOLD = 25
MIN_CONTOUR_AREA = 250

CAM_INDEX = 1

# TOP      -> black side on top
# BOTTOM   -> white side on top
# SIDE_L   -> left side view
# SIDE_R   -> right side view
BOARD_ORIENTATION = "TOP"

DEBUG_MODE = False


# =========================
# START ENGINE
# =========================
if not os.path.exists(ENGINE_PATH):
    print(f"[ERROR] Engine file not found: {ENGINE_PATH}")
    sys.exit(1)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

print(f"[INFO] Stockfish loaded from {ENGINE_PATH}")


# =========================
# LOAD CALIBRATION
# =========================
if not os.path.exists(CALIB_JSON):
    print(f"[ERROR] Calibration file not found: {CALIB_JSON}")
    engine.quit()
    sys.exit(1)

with open(CALIB_JSON, "r") as f:
    sq_points = json.load(f)

print(f"[INFO] Loaded {len(sq_points)} board squares")


# =========================
# BOARD ORIENTATION
# =========================
files = 'abcdefgh'
ranks = '12345678'


def remap_square(square_name: str):

    f = square_name[0]
    r = square_name[1]

    fi = files.index(f)
    ri = ranks.index(r)

    if BOARD_ORIENTATION == "TOP":
        return square_name

    elif BOARD_ORIENTATION == "BOTTOM":
        return f"{files[7-fi]}{ranks[7-ri]}"

    elif BOARD_ORIENTATION == "SIDE_L":
        return f"{files[ri]}{ranks[7-fi]}"

    elif BOARD_ORIENTATION == "SIDE_R":
        return f"{files[7-ri]}{ranks[fi]}"

    return square_name


# =========================
# HELPER FUNCTIONS
# =========================
def poly_center(pts):

    a = np.array(pts, np.int32)

    M = cv2.moments(a)

    if M["m00"] == 0:
        return int(a[:,0].mean()), int(a[:,1].mean())

    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])


def find_square(x, y):

    pt = (float(x), float(y))

    for sq, pts in sq_points.items():

        poly = np.array(pts, np.int32)

        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return sq

    return None


def overlay_poly(frame, poly_pts, color, alpha=0.45):

    overlay = frame.copy()

    pts = np.array(poly_pts, np.int32)

    cv2.fillPoly(overlay, [pts], color)

    return cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1-alpha,
        0
    )


# =========================
# DRAW BOARD LABELS
# =========================
def draw_board_labels(base_frame):

    overlay = base_frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    for sq, pts in sq_points.items():

        p = np.array(pts, np.int32)

        # draw polygon
        cv2.polylines(
            overlay,
            [p],
            True,
            (255,255,255),
            1
        )

        # draw coordinate label
        cx, cy = poly_center(pts)

        mapped = remap_square(sq)

        cv2.putText(
            overlay,
            mapped,
            (cx-12, cy+5),
            font,
            0.45,
            (0,255,255),
            1,
            cv2.LINE_AA
        )

    return overlay


# =========================
# SHOW DIGITAL BOARD
# =========================
def show_board(board, last_move=None):

    svg = chess.svg.board(
        board=board,
        lastmove=last_move,
        coordinates=True,
        size=450
    )

    png_data = cairosvg.svg2png(
        bytestring=svg.encode('utf-8')
    )

    img = Image.open(io.BytesIO(png_data))

    img_cv = cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow("Board State", img_cv)

    cv2.waitKey(1)


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    engine.quit()
    sys.exit(1)


# =========================
# GAME STATE
# =========================
board = chess.Board()

ref_frame = None

last_move = None

comp_turn = False

move_history = []


print("\n========== CONTROLS ==========")
print("r -> save before/after move")
print("u -> undo last move")
print("U -> undo last 2 moves")
print("d -> toggle debug")
print("q -> quit")
print("================================\n")

show_board(board)


# =========================
# MAIN LOOP
# =========================
try:

    while not board.is_game_over():

        ret, frame_raw = cap.read()

        if not ret:
            continue

        display = draw_board_labels(frame_raw.copy())

        cv2.imshow("Chess Tracker", display)

        key = cv2.waitKey(1) & 0xFF


        # =========================
        # TOGGLE DEBUG
        # =========================
        if key == ord('d'):

            DEBUG_MODE = not DEBUG_MODE

            state = "ON" if DEBUG_MODE else "OFF"

            print(f"[INFO] Debug mode: {state}")


        # =========================
        # PLAYER MOVE
        # =========================
        if key == ord('r'):

            # first press
            if ref_frame is None:

                ref_frame = frame_raw.copy()

                print("[DEBUG] Initial frame saved.")

            # second press
            else:

                print("[DEBUG] Final frame captured.")

                # =========================
                # IMAGE DIFFERENCE
                # =========================
                g1 = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)

                g1 = cv2.GaussianBlur(g1, (5,5), 0)
                g2 = cv2.GaussianBlur(g2, (5,5), 0)

                diff = cv2.absdiff(g1, g2)

                _, diff_thresh = cv2.threshold(
                    diff,
                    MOVE_THRESHOLD,
                    255,
                    cv2.THRESH_BINARY
                )

                diff_thresh = cv2.dilate(
                    diff_thresh,
                    None,
                    iterations=3
                )

                contours, _ = cv2.findContours(
                    diff_thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                detected = set()

                debug_frame = frame_raw.copy()

                # =========================
                # DETECT MOVED SQUARES
                # =========================
                for c in contours:

                    area = cv2.contourArea(c)

                    if area < MIN_CONTOUR_AREA:
                        continue

                    x, y, w, h = cv2.boundingRect(c)

                    cx = x + w//2
                    cy = y + h//2

                    sq = find_square(cx, cy)

                    if sq:

                        detected.add(sq)

                        # debug visualization
                        cv2.rectangle(
                            debug_frame,
                            (x,y),
                            (x+w,y+h),
                            (0,255,0),
                            2
                        )

                        cv2.circle(
                            debug_frame,
                            (cx,cy),
                            4,
                            (0,0,255),
                            -1
                        )

                        cv2.putText(
                            debug_frame,
                            sq,
                            (x,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2
                        )

                if DEBUG_MODE:
                    cv2.imshow("Detection Debug", debug_frame)

                print(f"[DEBUG] Detected squares: {detected}")


                # =========================
                # INTERPRET MOVE
                # =========================
                from_sq = None
                to_sq = None

                if len(detected) == 2:

                    a, b = list(detected)

                    piece_a = board.piece_at(
                        chess.parse_square(a)
                    )

                    piece_b = board.piece_at(
                        chess.parse_square(b)
                    )

                    if piece_a and not piece_b:
                        from_sq, to_sq = a, b

                    elif piece_b and not piece_a:
                        from_sq, to_sq = b, a

                    else:
                        from_sq, to_sq = a, b

                else:
                    print("[WARNING] Could not determine move.")


                # =========================
                # EXECUTE MOVE
                # =========================
                if from_sq and to_sq:

                    move = from_sq + to_sq

                    try:

                        mv = chess.Move.from_uci(move)

                        if mv in board.legal_moves:

                            board.push(mv)

                            move_history.append(mv)

                            last_move = mv

                            print(f"[YOU] Your move: {move}")

                            show_board(board, last_move)

                            # highlight move
                            frame_high = overlay_poly(
                                frame_raw.copy(),
                                sq_points[from_sq],
                                (0,255,0),
                                0.5
                            )

                            frame_high = overlay_poly(
                                frame_high,
                                sq_points[to_sq],
                                (0,0,255),
                                0.5
                            )

                            frame_high = draw_board_labels(frame_high)

                            cv2.imshow(
                                "Chess Tracker",
                                frame_high
                            )

                            cv2.waitKey(700)

                            comp_turn = True

                        else:
                            print(f"[WARNING] Illegal move: {move}")

                    except Exception as e:
                        print(f"[ERROR] Move error: {e}")

                ref_frame = None


        # =========================
        # UNDO LAST MOVE
        # =========================
        if key == ord('u'):

            if move_history:

                mv = move_history.pop()

                board.pop()

                print(f"[UNDO] Removed move: {mv}")

                show_board(board)

            else:
                print("[INFO] No moves to undo.")


        # =========================
        # UNDO LAST 2 MOVES
        # =========================
        if key == ord('U'):

            if len(move_history) >= 2:

                mv2 = move_history.pop()
                mv1 = move_history.pop()

                board.pop()
                board.pop()

                print(f"[UNDO] Removed: {mv1}, {mv2}")

                show_board(board)

            else:
                print("[INFO] Not enough moves.")


        # =========================
        # AI MOVE
        # =========================
        if comp_turn:

            result = engine.play(
                board,
                chess.engine.Limit(
                    time=random.uniform(0.4, 0.9)
                )
            )

            mv = result.move

            board.push(mv)

            move_history.append(mv)

            last_move = mv

            print(f"[AI] Computer move: {mv.uci()}")

            show_board(board, last_move)

            # highlight AI move
            move_str = mv.uci()

            frame_ai = overlay_poly(
                frame_raw.copy(),
                sq_points[move_str[:2]],
                (0,255,255),
                0.45
            )

            frame_ai = overlay_poly(
                frame_ai,
                sq_points[move_str[2:]],
                (0,165,255),
                0.45
            )

            frame_ai = draw_board_labels(frame_ai)

            cv2.imshow("Chess Tracker", frame_ai)

            cv2.waitKey(900)

            comp_turn = False


        # =========================
        # QUIT
        # =========================
        if key == ord('q'):

            print("[INFO] Exiting.")

            break


    print("[INFO] Game finished.")


finally:

    cap.release()

    cv2.destroyAllWindows()

    engine.quit()
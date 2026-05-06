import cv2
import numpy as np
import os

# ======================
# LOAD TEMPLATES
# ======================
def load_templates():
    templates = {}

    base_path = os.path.dirname(os.path.dirname(__file__))
    pieces_path = os.path.join(base_path, "pieces")

    names = ["wp","wr","wn","wb","wq","wk",
             "bp","br","bn","bb","bq","bk"]

    for name in names:
        path = os.path.join(pieces_path, f"{name}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception(f"Missing {path}")

        img = cv2.resize(img, (80, 80))
        templates[name] = img

    return templates


# ======================
# COLOR DETECTION (FIXED)
# ======================
def detect_color(img):
    img = cv2.resize(img, (80, 80))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # focus only center (ignore background/hand)
    center = gray[20:60, 20:60]
    mean_val = np.mean(center)

    # threshold (tune if needed)
    if mean_val > 140:
        return "w"
    else:
        return "b"


# ======================
# IDENTIFY PIECE
# ======================
def identify_piece(img, templates):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (80, 80))
    gray = cv2.equalizeHist(gray)

    color = detect_color(img)

    best = ""
    best_score = -1

    for name, tmpl in templates.items():
        if name[0] != color:
            continue

        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)

        if score > best_score:
            best_score = score
            best = name

    if best_score < 0.65:
        return ""

    return best


# ======================
# FORMAT LABEL
# ======================
def format_label(piece):
    if piece == "":
        return "No piece"

    color = "White" if piece[0] == "w" else "Black"

    names = {
        "p": "Pawn",
        "r": "Rook",
        "n": "Knight",
        "b": "Bishop",
        "q": "Queen",
        "k": "King"
    }

    return f"{color} {names[piece[1]]}"



# ======================
# INIT
# ======================
templates = load_templates()
cap = cv2.VideoCapture(1)  # try 1 if needed


# ======================
# LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ===== CENTER BOX =====
    box_size = 220
    cx, cy = w // 2, h // 2

    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    roi = frame[y1:y2, x1:x2]

    # ===== DETECT =====
    piece = identify_piece(roi, templates)
    label = format_label(piece)

    # ===== DRAW BOX =====
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # ===== LABEL =====
    cv2.rectangle(frame, (x1, y1-40), (x2, y1), (0,255,0), -1)

    cv2.putText(frame, label,
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,0), 2)

    cv2.imshow("Chess Piece Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print("Point:", x, y)

cap = cv2.VideoCapture(1)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    if len(points) == 4:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---- check ----
if len(points) != 4:
    print("Need 4 points!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# ---- reorder points ----
pts = np.array(points, dtype="float32")

s = pts.sum(axis=1)
diff = np.diff(pts, axis=1)

top_left = pts[np.argmin(s)]
bottom_right = pts[np.argmax(s)]
top_right = pts[np.argmin(diff)]
bottom_left = pts[np.argmax(diff)]

pts1 = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

pts2 = np.float32([
    [0, 0],
    [640, 0],
    [0, 640],
    [640, 640]
])

# ---- transform ----
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(frame, matrix, (640, 640))

# ---- crop fix ----
result = result[20:620, 20:620]
result = cv2.resize(result, (640, 640))

# ---- grid ----
for i in range(1, 8):
    cv2.line(result, (0, i*80), (640, i*80), (0, 255, 0), 1)
    cv2.line(result, (i*80, 0), (i*80, 640), (0, 255, 0), 1)

# ---- ASK USER WHAT TO SAVE ----
choice = input("Type 'e' for empty board or 'c' for current board: ")

if choice == 'e':
    cv2.imwrite("empty_board.jpg", result)
    print("Saved empty_board.jpg ✅")

elif choice == 'c':
    cv2.imwrite("fixed_board.jpg", result)
    print("Saved fixed_board.jpg ✅")

# save matrix once
np.save("matrix.npy", matrix)

cv2.imshow("Fixed Board", result)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
import cv2

cap = cv2.VideoCapture(1)   # use your external cam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to square
    frame = cv2.resize(frame, (640, 640))

    # Draw 8x8 grid
    for i in range(1, 8):
        cv2.line(frame, (0, i*80), (640, i*80), (0, 255, 0), 1)
        cv2.line(frame, (i*80, 0), (i*80, 640), (0, 255, 0), 1)

    cv2.imshow("Grid Check", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
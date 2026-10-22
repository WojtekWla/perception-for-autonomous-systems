import cv2

cap = cv2.VideoCapture('rolling_ball_challenge.mp4')
hsv = None

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV at", (x, y), "=", hsv[y, x])

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', on_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
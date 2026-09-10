import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


DRAW_LINES = False

drawing = False
ix, iy = -1,-1
rect = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img.copy()
            cv2.rectangle(img_display, (ix, iy), (x,y), (0,255,0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        img_display = img.copy()
        cv2.rectangle(img_display, (ix, iy), (x,y), (0,255,0), 2)




cap = cv2.VideoCapture('data/Challenge.mp4')

ret, img = cap.read()

x1, y1, x2, y2 = -1,-1,-1,-1
corners = None
gray1 = None
if ret:
    gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray1, 200, 0.01, minDistance=10)


    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)

    img_display = img.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)
    while True:
        cv2.imshow("Image", img_display)
        if drawing == False and rect:
            x1, y1, x2, y2 = rect
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    pts = corners.reshape(-1, 2)
    mask = (pts[:, 0] >= x1) & (pts[:, 1] >= y1) &\
        (pts[:, 0] <= x2) & (pts[:, 1] <= y2)

    corners = corners[mask] 

print(x1, y1, x2, y2)

trail_mask = np.zeros_like(img)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    new_corners, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)

    good = status.ravel() == 1
    old_pts = corners[good]
    new_pts = new_corners[good]

    MOVE_THRESHOLD = 2.0
    for i in range(len(new_pts)):
        x_old, y_old = old_pts[i][0]
        x_new, y_new = new_pts[i][0]
        dist = np.hypot(x_new - x_old, y_new - y_old)
        if DRAW_LINES:
            cv2.line(trail_mask, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 255, 0), 2)
        if dist > MOVE_THRESHOLD:
            cv2.circle(frame, (int(x_new), int(y_new)), 10, (0, 255, 0), -1)

    corners = np.array(new_pts, dtype=np.float32)

    gray1 = gray2
    result = cv2.add(frame, trail_mask)
    cv2.imshow("Video", result)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import imutils

I = np.eye(6)

def update(x, P, Z, H, R):
    y = Z - np.dot(H, x)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    x = x + np.dot(K, y)
    P = np.dot((I - np.dot(K, H)), P)

    return [x, P]

def predict(x, P, F, u):
    x = np.dot(F, x) + u
    P = np.dot(F, np.dot(P, F.T))
    return [x, P]

cap = cv2.VideoCapture('rolling_ball.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

center, radius = (-1,-1),-1

#position_x, velocity_x, acceleration_x, position_y, velocity_y, acceleration_y
x = np.array([[0],[0],[0],[0],[0],[0]])
P = np.array([
    [1000, 0, 0, 0, 0, 0],
    [0, 1000, 0, 0, 0, 0],
    [0, 0, 1000, 0, 0, 0],
    [0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 1000, 0],
    [0, 0, 0, 0, 0, 1000],
])

u = np.array([[0],[0],[0],[0],[0],[0]])
F = np.array([[1, 1, 0.5, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0.5],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]]
)
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
])

R = np.array([
    [10, 0],
    [0, 10]
])

first, detected = True, False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    b,g,r = cv2.split(frame)
    image = cv2.merge([r,g,b])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask = cv2.bitwise_or(cv2.inRange(image, lower_red, upper_red), cv2.inRange(image, lower_red2, upper_red2))
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    output = frame.copy()
    if cnts:
        largest = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(largest) > 100:
            detected = True
            (cx, cy), new_radius = cv2.minEnclosingCircle(largest)
            center = (int(cx), int(cy))
            if radius == -1:
                radius = int(new_radius)

            # cv2.circle(output, center, radius, (0,0,0), 3)
            if first:
                x[0,0], x[3, 0] = cx, cy
                P[0,0], P[3,3] = 10, 10
                first = False
    if detected and not first:
        Z = np.array([[center[0]], [center[1]]])
        [x, P] = update(x, P, Z, H, R)
        cv2.circle(output, (int(x[0,0]), int(x[3,0])), radius, (0,255,0), 3)

    if not first:
        [x, P] = predict(x, P, F, u)
        cv2.circle(output, (int(x[0,0]), int(x[3,0])), radius, (255, 0, 0), 3)

    cv2.imshow('Frame', output)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    detected = False

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import imutils

I = np.eye(6) #identity matrix

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

x = np.array([[0],[0],[0],[0],[0],[0]]) #initial state, all zero since the ball hasn't been detected yet
P = np.array([ #uncertainty matrix: lower values mean more confidence, everything starts fully unknown (1000) until the first detection arrives
    [1000, 0, 0, 0, 0, 0],
    [0, 1000, 0, 0, 0, 0],
    [0, 0, 1000, 0, 0, 0],
    [0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 1000, 0],
    [0, 0, 0, 0, 0, 1000],
])

u = np.array([[0],[0],[0],[0],[0],[0]]) #external motion, assumed to be none so stays zero
F = np.array([[1, 1, 0.5, 0, 0, 0], #transition matrix, models constant acceleration motion: pos += vel + 0.5*accel, vel += accel, accel stays the same
              [0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0.5],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]]
)
H = np.array([ #observation matrix, maps the 6d state down to what is actually measured: position x and position y
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
])

R = np.array([ #measurement uncertainty for the position x and y measurements coming from the detection
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

    #defining boundaries for the red color in hsv
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    #creating mask
    mask = cv2.bitwise_or(cv2.inRange(image, lower_red, upper_red), cv2.inRange(image, lower_red2, upper_red2))
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    output = frame.copy()
    if cnts:
        largest = max(cnts, key=cv2.contourArea) #assuming the ball is the biggest red blob, bigger than any noise

        if cv2.contourArea(largest) > 100: #checking for the size
            detected = True
            (cx, cy), new_radius = cv2.minEnclosingCircle(largest)
            center = (int(cx), int(cy))
            if radius == -1: #updating radius only at the beginning 
                radius = int(new_radius)
            #if the ball was not detected previously we want to adjust it starting position and confidence
            if first:
                x[0,0], x[3, 0] = cx, cy #setting position x and y from the first detection
                P[0,0], P[3,3] = 10, 10 #position x and y become confident after detection
                first = False
    if detected and not first: #updating ball's position if it's not first detection
        Z = np.array([[center[0]], [center[1]]])
        [x, P] = update(x, P, Z, H, R) 
        cv2.circle(output, (int(x[0,0]), int(x[3,0])), radius, (0,255,0), 3)

    if not first: #predicting ball's position even if it is occluded
        [x, P] = predict(x, P, F, u)
        cv2.circle(output, (int(x[0,0]), int(x[3,0])), radius, (255, 0, 0), 3)

    cv2.imshow('Frame', output)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    detected = False

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import imutils

balls = []
MATCH_THRESHOLD = 50
MAX_MISSES = 15

class Ball:
    def __init__(self, cx, cy, radius):
        #position_x, velocity_x, acceleration_x, position_y, velocity_y, acceleration_y
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.matched = False
        self.misses = 0
        self.x = np.array([[cx],[0],[0],[cy],[0],[0]])
        self.P = np.array([
            [10, 0, 0, 0, 0, 0],
            [0, 1000, 0, 0, 0, 0],
            [0, 0, 1000, 0, 0, 0],
            [0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 1000, 0],
            [0, 0, 0, 0, 0, 1000],
        ])



I = np.eye(6)
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

def find_ball(cx, cy):
    def distances(point, points):
        return np.linalg.norm(np.asarray(points) - np.asarray(point), axis=1)

    if not balls:
        return None

    positions = np.array([[b.x[0,0], b.x[3,0]] for b in balls])
    d = distances((cx, cy), positions)
    idx = np.argmin(d)

    if d[idx] > MATCH_THRESHOLD:
        return None
    
    balls[idx].matched = True
    return balls[idx]


cap = cv2.VideoCapture('../data/week_7/rolling_ball_challenge.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

detected = False

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

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.bitwise_or(cv2.inRange(image, lower_red, upper_red), cv2.inRange(image, lower_red2, upper_red2))
    mask = cv2.bitwise_or(mask, cv2.inRange(image, lower_blue, upper_blue))
    mask = cv2.bitwise_or(mask, cv2.inRange(image, lower_yellow, upper_yellow))
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    output = frame.copy()
    ball = None
    if cnts:
        for cnt in cnts:
            if cv2.contourArea(cnt) > 100:
                detected = True
                (cx, cy), new_radius = cv2.minEnclosingCircle(cnt)
                center = (int(cx), int(cy))
                radius = int(new_radius)

                ball = find_ball(cx, cy)
                if not ball:
                    ball = Ball(cx, cy, radius)
                    ball.matched = True
                    balls.append(ball)
                else:
                    Z = np.array([[center[0]], [center[1]]])
                    [ball.x, ball.P] = update(ball.x, ball.P, Z, H, R)
                    cv2.circle(output, (int(ball.x[0,0]), int(ball.x[3,0])), ball.radius, (0,255,0), 3)

    for ball in balls[:]:
        [ball.x, ball.P] = predict(ball.x, ball.P, F, u)

        if ball.matched:
            ball.misses = 0
        else:
            ball.misses += 1

        if ball.misses > MAX_MISSES:
            balls.remove(ball)
        else:
            cv2.circle(output, (int(ball.x[0,0]), int(ball.x[3,0])), ball.radius, (255, 0, 0), 3)

    for ball in balls:
        ball.matched = False

    cv2.imshow('Frame', output)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    detected = False

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import math

# ---------- Camera and hand tracking ----------
hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AMBER = (0, 184, 255)
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
LM_STYLE = draw.DrawingSpec(color=AMBER, thickness=2, circle_radius=2)
CONN_STYLE = draw.DrawingSpec(color=WHITE, thickness=1)

TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]

# ---------- Pinch settings (ratio of hand size, with hysteresis) ----------
PINCH_ON = 0.35
PINCH_OFF = 0.50
pinched = False

# ---------- Motion smoothing ----------
SMOOTH = 0.5          # 0 = no smoothing, closer to 1 = very smooth but laggy
prev_pos = None       # last palm position
prev_gesture = None
dx_s = dy_s = 0.0     # smoothed movement


def dist(p, a, b):
    return math.hypot(p[a].x - p[b].x, p[a].y - p[b].y)


def finger_states(lm):
    p = lm.landmark
    thumb = dist(p, 4, 17) > dist(p, 3, 17)
    others = [p[t].y < p[k].y for t, k in zip(TIPS, PIPS)]
    return tuple(int(v) for v in [thumb] + others)   # thumb, index, middle, ring, pinky


def pinch_ratio(lm):
    p = lm.landmark
    size = dist(p, 0, 9)
    return dist(p, 4, 8) / size if size > 1e-6 else 9.9


def classify(lm):
    """Return the gesture name for this frame."""
    global pinched
    ratio = pinch_ratio(lm)
    pinched = ratio < (PINCH_OFF if pinched else PINCH_ON)
    if pinched:
        return "PINCH"

    s = finger_states(lm)
    thumb, rest = s[0], s[1:]
    if rest == (0, 0, 0, 0) and not thumb:
        return "REST"            # fist
    if rest == (1, 0, 0, 0):
        return "ROTATE" if thumb else "POINT"
    if rest == (1, 1, 0, 0):
        return "ZOOM"
    if rest == (1, 1, 1, 1):
        return "PAN"             # open hand
    return "UNKNOWN"


def label_box(frame, text, x, y):
    h, w, _ = frame.shape
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x = max(4, min(x, w - tw - 8))
    y = max(th + 8, min(y, h - 4))
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), (30, 30, 30), -1)
    cv2.putText(frame, text, (x, y), FONT, 0.6, WHITE, 2)


# What each gesture does with the palm movement (hook the real actions in here)
def act(gesture, dx, dy):
    if gesture == "ROTATE":
        pass    # rotate model: yaw += dx, pitch += dy
    elif gesture == "ZOOM":
        pass    # zoom += -dy
    elif gesture == "PAN":
        pass    # camera.x += dx, camera.y += dy
    elif gesture == "PINCH":
        pass    # drag selected object by (dx, dy)
    elif gesture == "POINT":
        pass    # cursor position = prev_pos


# ---------- Main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w, _ = frame.shape

    gesture = None
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        gesture = classify(lm)

        # palm centre = middle finger knuckle (steady, doesn't jump with finger poses)
        pos = (lm.landmark[9].x, lm.landmark[9].y)

        # only measure movement if the gesture didn't just change (avoids jumps)
        if prev_pos is not None and gesture == prev_gesture:
            raw_dx = pos[0] - prev_pos[0]
            raw_dy = pos[1] - prev_pos[1]
            dx_s = SMOOTH * dx_s + (1 - SMOOTH) * raw_dx
            dy_s = SMOOTH * dy_s + (1 - SMOOTH) * raw_dy
            act(gesture, dx_s, dy_s)
        else:
            dx_s = dy_s = 0.0

        prev_pos = pos
        prev_gesture = gesture

        draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=LM_STYLE,
                            connection_drawing_spec=CONN_STYLE)
        x = int(lm.landmark[0].x * w) - 30
        y = int(lm.landmark[0].y * h) + 35
        label_box(frame, f"{gesture}  dx={dx_s:+.3f} dy={dy_s:+.3f}", x, y)
    else:
        prev_pos = None
        prev_gesture = None
        pinched = False
        dx_s = dy_s = 0.0

    cv2.putText(frame, gesture or "no hand", (14, h - 16), FONT, 1.0,
                GREEN if gesture == "PINCH" else AMBER, 2)
    cv2.imshow("hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:     # Esc
        break

cap.release()
cv2.destroyAllWindows()
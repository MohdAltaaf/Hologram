"""
Hand gestures -> WebSocket (ws://localhost:8765) -> viewer.html

Install:  pip install websockets
Run:      python gestures_ws.py   then open viewer.html in Chrome/Chromium
Keys (camera window):  P = toggle preview (off = faster)   Esc = quit
"""
import asyncio
import json
import math
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import websockets

# ---------- WebSocket server (runs in a background thread) ----------
clients = set()
loop = asyncio.new_event_loop()


async def handler(ws):
    clients.add(ws)
    print(f"viewer connected ({len(clients)})")
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"viewer disconnected ({len(clients)})")


async def serve():
    async with websockets.serve(handler, "localhost", 8765, ping_interval=None):
        await asyncio.Future()


def run_server():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve())


threading.Thread(target=run_server, daemon=True).start()


def broadcast(msg):
    if clients:
        loop.call_soon_threadsafe(websockets.broadcast, set(clients), msg)


# ---------- Camera on its own thread (always holds the NEWEST frame) ----------
class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # faster than raw video
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.count = 0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.count += 1

    def read(self):
        with self.lock:
            return self.frame, self.count

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ---------- One Euro filter: smooth when slow, responsive when fast ----------
class OneEuro:
    def __init__(self, mincutoff=1.0, beta=8.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1 - a) * self.x_prev
        self.t_prev = t
        return self.x_prev


cursor_fx, cursor_fy = OneEuro(1.5, 10.0), OneEuro(1.5, 10.0)   # index fingertip cursor
palm_fx, palm_fy = OneEuro(1.0, 8.0), OneEuro(1.0, 8.0)         # palm, for movement deltas

# ---------- Hand tracking ----------
hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    model_complexity=0,          # try 1 for better accuracy if your PC can take it
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
draw = mp.solutions.drawing_utils

FONT = cv2.FONT_HERSHEY_SIMPLEX
AMBER = (0, 184, 255)
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
LM_STYLE = draw.DrawingSpec(color=AMBER, thickness=2, circle_radius=2)
CONN_STYLE = draw.DrawingSpec(color=WHITE, thickness=1)

# (knuckle, middle joint, tip) for index, middle, ring, pinky
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]

# ---------- Tunables ----------
PINCH_ON = 0.35
PINCH_OFF = 0.50
THUMB_OUT = 0.60       # thumb tip to index knuckle / hand size; above = thumb sticks out
EXTEND_ANGLE = 125     # finger counts as extended if its middle joint is straighter than this (degrees)
STABLE_FRAMES = 3      # a new gesture must hold this many frames before it counts

pinched = False
stable_g = None
cand_g = None
cand_count = 0
prev_pos = None
prev_gesture = None
dx_f = dy_f = 0.0
thumb_ratio = 0.0
debug_angles = (0, 0, 0, 0)


def dist(p, a, b):
    return math.hypot(p[a].x - p[b].x, p[a].y - p[b].y)


def joint_angle(p, a, b, c):
    """Angle at joint b between a and c, in degrees (3D, so hand rotation doesn't matter)."""
    v1 = (p[a].x - p[b].x, p[a].y - p[b].y, p[a].z - p[b].z)
    v2 = (p[c].x - p[b].x, p[c].y - p[b].y, p[c].z - p[b].z)
    n1 = math.sqrt(sum(v * v for v in v1))
    n2 = math.sqrt(sum(v * v for v in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosv = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def classify(lm):
    """Raw gesture for one frame."""
    global pinched, thumb_ratio, debug_angles
    p = lm.landmark
    size = dist(p, 0, 9)
    if size < 1e-6:
        return "UNKNOWN"

    pr = dist(p, 4, 8) / size
    pinched = pr < (PINCH_OFF if pinched else PINCH_ON)
    if pinched:
        return "PINCH"

    angles = tuple(joint_angle(p, a, b, c) for a, b, c in FINGERS)
    debug_angles = tuple(int(a) for a in angles)
    rest = tuple(int(a > EXTEND_ANGLE) for a in angles)   # index, middle, ring, pinky

    thumb_ratio = dist(p, 4, 5) / size
    thumb_out = thumb_ratio > THUMB_OUT

    if rest == (0, 0, 0, 0):
        return "REST"
    if rest == (1, 0, 0, 0):
        return "ROTATE" if thumb_out else "POINT"
    if rest == (1, 1, 0, 0):
        return "ZOOM"
    if rest == (1, 1, 1, 1):
        return "PAN"
    return "UNKNOWN"


def debounce(raw):
    global stable_g, cand_g, cand_count
    if raw == stable_g:
        cand_g, cand_count = None, 0
    elif raw == cand_g:
        cand_count += 1
        if cand_count >= STABLE_FRAMES:
            stable_g, cand_g, cand_count = raw, None, 0
    else:
        cand_g, cand_count = raw, 1
    return stable_g


def label_box(frame, text, x, y):
    h, w, _ = frame.shape
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x = max(4, min(x, w - tw - 8))
    y = max(th + 8, min(y, h - 4))
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), (30, 30, 30), -1)
    cv2.putText(frame, text, (x, y), FONT, 0.6, WHITE, 2)


# ---------- Main loop ----------
cam = Camera(0)
show_preview = True
last_count = -1
fps = 0.0
t_fps = time.perf_counter()

while True:
    frame, count = cam.read()
    if frame is None or count == last_count:
        time.sleep(0.001)           # no new frame yet
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
    last_count = count

    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_fps, 1e-3))
    t_fps = now

    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w, _ = frame.shape

    gesture = None
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        gesture = debounce(classify(lm))

        # filtered palm position -> movement since last frame
        px = palm_fx(lm.landmark[9].x, now)
        py = palm_fy(lm.landmark[9].y, now)
        if prev_pos is not None and gesture == prev_gesture:
            dx_f, dy_f = px - prev_pos[0], py - prev_pos[1]
        else:
            dx_f = dy_f = 0.0
        prev_pos = (px, py)
        prev_gesture = gesture

        # filtered index fingertip -> cursor
        tip = lm.landmark[8]
        cx = cursor_fx(tip.x, now)
        cy = cursor_fy(tip.y, now)

        broadcast(json.dumps({
            "g": gesture,
            "dx": round(dx_f, 5),
            "dy": round(dy_f, 5),
            "x": round(cx, 4),
            "y": round(cy, 4),
        }))

        if show_preview:
            draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=LM_STYLE,
                                connection_drawing_spec=CONN_STYLE)
            x = int(lm.landmark[0].x * w) - 30
            y = int(lm.landmark[0].y * h) + 35
            label_box(frame, f"{gesture}  dx={dx_f:+.3f} dy={dy_f:+.3f}", x, y)
            cv2.putText(frame, f"angles {debug_angles}  thumb {thumb_ratio:.2f}", (14, 24),
                        FONT, 0.55, WHITE, 1)
    else:
        stable_g = cand_g = None
        cand_count = 0
        prev_pos = None
        prev_gesture = None
        pinched = False
        dx_f = dy_f = 0.0
        for f in (cursor_fx, cursor_fy, palm_fx, palm_fy):
            f.reset()
        broadcast(json.dumps({"g": None, "dx": 0, "dy": 0, "x": 0.5, "y": 0.5}))

    if show_preview:
        cv2.putText(frame, gesture or "no hand", (14, h - 16), FONT, 1.0,
                    GREEN if gesture == "PINCH" else AMBER, 2)
        cv2.putText(frame, f"{fps:.0f} fps", (w - 100, h - 16), FONT, 0.6, WHITE, 1)
        cv2.imshow("hands", frame)
    else:
        status = np.zeros((60, 320, 3), np.uint8)
        cv2.putText(status, f"{gesture or 'no hand'}   {fps:.0f} fps", (10, 28), FONT, 0.7, AMBER, 2)
        cv2.putText(status, "P: preview   Esc: quit", (10, 50), FONT, 0.45, WHITE, 1)
        cv2.imshow("hands", status)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord("p"), ord("P")):
        show_preview = not show_preview
        cv2.destroyWindow("hands")

cam.release()
cv2.destroyAllWindows()
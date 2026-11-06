# Holo CAD

A floating, hand-controlled 3D modeler. You sketch, extrude, revolve and fillet parts with your bare hand, and the result is shown as a **real aerial image** floating in the air in front of you, not behind a screen.

Inspired by the holographic design scenes in Iron Man.

> **Status:** the hand tracking, gesture set and CAD viewer work on a normal screen. The optical display (screen + beam splitter + retroreflector) is planned and has not been built yet.

---

## How it works

```
 webcam ──> gestures_ws.py ──WebSocket──> cad.html (three.js)
            MediaPipe hand tracking        sketch / extrude / revolve / fillet
            gesture classification         shown on a screen, later floated in the air
```

1. `gestures_ws.py` reads the webcam, tracks one hand with MediaPipe, classifies a gesture each frame, and streams `{gesture, dx, dy, x, y}` over a local WebSocket (`ws://localhost:8765`).
2. `cad.html` listens to that stream, drives a cursor and the modeling tools, and renders everything with three.js.

## The display idea

Most "holograms" are a screen reflected in a tilted sheet, which only looks like it is in the room. This project forms a **real image**: light rays actually converge at a point in free space.

1. A bright screen emits light.
2. A beam splitter reflects part of it onto a retroreflector.
3. The retroreflector sends the light back along the path it came from.
4. The light passes through the beam splitter and converges into a floating 2D image, the same size as the screen, mirrored across the beam splitter plane.
<img width="3000" height="4000" alt="image" src="https://github.com/user-attachments/assets/e018f99a-0f85-4fcf-b7ba-e5b53980721c" />


## Requirements

- Python 3.9 to 3.12
- A webcam
- A browser with internet access as it downloads fonts on load 

```
pip install mediapipe==0.10.14 opencv-python websockets numpy
```

The code uses MediaPipe's `mp.solutions.hands` API. Newer MediaPipe releases removed it, which is why the version is pinned.
ggles the preview (off is faster), `Esc` quits.

## Gestures

| Gesture | Pose | Use |
|---|---|---|
| Point | index finger only | move the cursor |
| Pinch | thumb and index tips together | click, place a point, hold and drag to change a value |
| Rotate | thumb and index out, others folded | orbit the part (feature mode) |
| Zoom | index and middle up | hand up or down zooms |
| Pan | open hand | slide the view |
| Rest | fist | do nothing |

The cursor follows your index fingertip.
<img width="925" height="647" alt="image" src="https://github.com/user-attachments/assets/1ddd59db-0b27-4217-ac31-ddeac1147c9e" />


## Modeling workflow

1. **Sketch** on the grid: Rectangle (two corners), Circle (center, then an edge point) or Polygon (points, then the first point or Close).
2. **Extrude** or **Revolve** the closed profile and watch the live orange preview.
3. Adjust **Depth** (out, in, or Symmetric for both directions), **Angle** (revolve, 15 to 360 degrees) and **Fillet** (rounds corners and cap edges). Pinch and drag up or down, or use the `-` and `+` buttons.
4. **Accept** to keep the part or **Cancel**.
5. **Export STL** saves everything as `part.stl` for slicing or other CAD tools.

<img width="1917" height="895" alt="image" src="https://github.com/user-attachments/assets/e77db1c8-cac9-4401-aab5-9a0950bc3159" />


Revolve spins the profile around the green Y axis, so keep the sketch on the right side of it.

### Built for imperfect tracking

- **Snap grid** (0.25, 0.5 or 1 unit). The snap marker only moves to a new node once the cursor is clearly past the current one.
- **Click where you aimed:** pinching moves the fingertip, so a click lands where the cursor was about 180 ms earlier (`REWIND`).
- **Dwell click** (off by default): hold still on a toolbar button to click it. Never places sketch points.
- Toolbar buttons never move between modes; unavailable ones are dimmed.
- If the hand is lost, the cursor stays put and work in progress is not cancelled.
- Undo and redo for sketch points and parts.

### Keyboard

| Key | Action |
|---|---|
| `Z` / `Y` | undo / redo |
| `E` | export STL |
| `M` | mirror the page (for the optics) |
| `F` | fullscreen |
| `H` | hide the interface |
| `Enter` / `Esc` | accept / cancel a feature |
| `+` / `-` | change the active value |



## Roadmap

- [ ] Build and test the optical display
- [ ] Calibration step that maps your reachable hand area to the whole screen
- [ ] Move and select finished parts, load STL/OBJ files
- [ ] Boolean add and subtract, two-hand scaling
- [ ] Depth "poke" click with a time-of-flight sensor
- [ ] Run everything on a Raspberry Pi 4 or 5 for a self-contained build

## Credits

Built by Mohd Altaaf for Fallen - Hackclub.

## License

MIT License

"""
Traffic-light state and stop signs from the front camera (no simulator state).

A pretrained COCO YOLO (no training; the same ultralytics yolov8n the env's
perception pipeline already uses) finds 'traffic light' (class 9) and
'stop sign' (class 11) boxes.  The light that governs us is taken to be the
largest light box in the central upper part of the image; its colour comes
from counting bright red / yellow / green pixels in HSV.  A short majority
vote over the last frames removes single-frame flicker.

Where to stop is decided by the caller without a map: at the next junction
entry on the route (the leaderboard's route commands mark junctions).
"""

from collections import Counter, deque


TL_CLASS, STOP_CLASS = 9, 11


def classify_light(crop_bgr):
    """'red' / 'yellow' / 'green' / 'unknown' from a light's crop (BGR)."""
    import cv2
    if crop_bgr is None or crop_bgr.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (v > 130) & (s > 80)
    red = lit & ((h < 10) | (h > 165))
    yellow = lit & (h >= 15) & (h <= 35)
    green = lit & (h >= 40) & (h <= 95)
    counts = {'red': int(red.sum()), 'yellow': int(yellow.sum()), 'green': int(green.sum())}
    best = max(counts, key=counts.get)
    if counts[best] < max(4, 0.02 * crop_bgr.shape[0] * crop_bgr.shape[1]):
        return 'unknown'
    return best


class TrafficLightDetector:
    # Region of the 1600x900 front image searched for lights / signs: lights
    # 40 m away are ~10 px tall at full resolution, ~4 px if the whole image is
    # shrunk to YOLO's 640 -- so crop first, then run at higher resolution.
    ROI = (0.2, 0.8, 0.0, 0.7)      # x0, x1, y0, y1 as fractions

    def __init__(self, model='yolov8n.pt', conf=0.25, imgsz=960, vote=5):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf, self.imgsz = conf, imgsz
        self.history = deque(maxlen=vote)
        self.last = {}

    def detect(self, img_bgr):
        """-> {'light': red/yellow/green/unknown/None (no light), 'stop_sign':
        bool, 'stop_sign_frac': box area / image area}."""
        H, W = img_bgr.shape[:2]
        rx0, rx1, ry0, ry1 = (int(self.ROI[0] * W), int(self.ROI[1] * W),
                              int(self.ROI[2] * H), int(self.ROI[3] * H))
        roi = img_bgr[ry0:ry1, rx0:rx1]
        res = self.model.predict(roi, imgsz=self.imgsz, conf=self.conf,
                                 classes=[TL_CLASS, STOP_CLASS], verbose=False)[0]
        light_box, stop_frac = None, 0.0
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            x0, y0, x1, y1 = b + [rx0, ry0, rx0, ry0]       # back to full-image pixels
            cx, cy, area = 0.5 * (x0 + x1), 0.5 * (y0 + y1), (x1 - x0) * (y1 - y0)
            if int(c) == STOP_CLASS:
                stop_frac = max(stop_frac, area / (W * H))
            elif abs(cx - W / 2) < 0.35 * W and cy < 0.6 * H and (y1 - y0) > 8:
                if light_box is None or area > light_box[4]:
                    light_box = (int(x0), int(y0), int(x1), int(y1), area)
        if light_box is None:
            state = None
        else:
            x0, y0, x1, y1, _ = light_box
            state = classify_light(img_bgr[max(y0, 0):y1, max(x0, 0):x1])
        self.history.append(state)
        votes = Counter(s for s in self.history if s is not None)
        voted = votes.most_common(1)[0][0] if votes else None
        self.last = dict(light=voted, raw=state, stop_sign=stop_frac > 0.002,
                         stop_sign_frac=stop_frac)
        return self.last

"""
Ego-lane centre from the front camera with a pretrained YOLOP (hustvl/YOLOP,
BDD100K drivable-area + lane-line segmentation; no training here), so the
agent needs only the leaderboard's SPARSE route -- like UniAD / VAD / ORION --
instead of the full-resolution lane-centre route.

    lanes = LanePerception('models/yolop/yolop-640-640.onnx')
    est = lanes.update(front_bgr)      # {'valid', 'coeffs', 'x_max', ...}

Pipeline:
  1. YOLOP ONNX, preprocessing exactly as the official test_onnx.py
     (letterbox to 640x640 with 114 padding, RGB, ImageNet mean/std)
  2. lane-line pixels -> ground plane with the known camera geometry (CARLA
     pinhole: fx = W / (2 tan(fov/2)), mounted at x, z, zero pitch): a pixel
     (u, v) below the horizon hits the road at forward X = z_cam * fx / (v - cy),
     right Y = X * (u - cx) / fx
  3. per 2 m forward bin, cluster the lateral positions into lines; the ego lane
     is bounded by the nearest line left and right of the car (centre =
     midpoint; with one line only, offset by half a lane)
  4. robust quadratic fit right(x) = c0 + c1 x + c2 x^2 over the bins

Coordinates: vehicle frame, x forward from the vehicle origin, y RIGHT
(CARLA / Unreal convention, the same as the route's Frenet d).
"""

import math

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def letterbox(img, new_shape=(640, 640), color=114):
    """YOLOP's resize_unscale (test_onnx.py)."""
    import cv2
    h, w = img.shape[:2]
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.float32)
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas[dh:dh + nh, dw:dw + nw, :] = img
    return canvas, r, dw, dh, nw, nh


class CameraGeometry:
    def __init__(self, W=1600, H=900, fov_deg=70.0, x=0.80, y=0.0, z=1.60):
        self.W, self.H = W, H
        self.f = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        self.cx, self.cy = W / 2.0, H / 2.0
        self.x, self.y, self.z = x, y, z

    def pixels_to_ground(self, u, v):
        """(N,) pixel coords -> (X forward, Y right) in the vehicle frame for
        pixels below the horizon; NaN elsewhere."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        dv = v - self.cy
        ok = dv > 1.0
        X = np.full(u.shape, np.nan)
        Y = np.full(u.shape, np.nan)
        X[ok] = self.z * self.f / dv[ok]
        Y[ok] = X[ok] * (u[ok] - self.cx) / self.f
        return X + self.x, Y + self.y

    def ground_to_pixels(self, X, Y):
        """Inverse (for tests): vehicle-frame ground point -> pixel."""
        Xc = np.asarray(X, float) - self.x
        u = self.cx + self.f * (np.asarray(Y, float) - self.y) / Xc
        v = self.cy + self.f * self.z / Xc
        return u, v


def fit_lane_centre(X, Y, x_bins=(4.0, 36.0, 2.0), lane_width=3.5, gap=0.8):
    """Lane-line ground points -> quadratic of the ego-lane centre, right(x)."""
    xs, cs, ws = [], [], []
    lo, hi, step = x_bins
    for x0 in np.arange(lo, hi, step):
        # Follow the lane outward: lines are searched around the centre
        # extrapolated from the bins so far, not around the car's axis -- on a
        # bend both lines lie on one side of y = 0 further out.
        xm = x0 + step / 2.0
        if len(xs) >= 2:
            ref = cs[-1] + (cs[-1] - cs[-2]) / (xs[-1] - xs[-2]) * (xm - xs[-1])
        elif xs:
            ref = cs[-1]
        else:
            ref = 0.0
        m = (X >= x0) & (X < x0 + step) & np.isfinite(Y) & (np.abs(Y) < 12.0)
        if m.sum() < 3:
            continue
        ys = np.sort(Y[m])
        # cluster lateral positions into individual lines
        groups, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] > gap:
                groups.append(cur)
                cur = [y]
            else:
                cur.append(y)
        groups.append(cur)
        lines = [float(np.median(g)) for g in groups if len(g) >= 2]
        left = [l for l in lines if l < ref]
        right = [l for l in lines if l > ref]
        yl = max(left) if left else None          # nearest line on the left
        yr = min(right) if right else None        # nearest line on the right
        if yl is not None and yr is not None and 2.5 < yr - yl < 5.0:
            c, w = 0.5 * (yl + yr), 1.0
        elif yl is not None and yl > ref - lane_width:
            c, w = yl + lane_width / 2.0, 0.5
        elif yr is not None and yr < ref + lane_width:
            c, w = yr - lane_width / 2.0, 0.5
        else:
            continue
        xs.append(x0 + step / 2.0)
        cs.append(c)
        ws.append(w)
    if len(xs) < 3:
        return None
    xs, cs, ws = np.array(xs), np.array(cs), np.array(ws)
    coeffs = None
    for _ in range(2):                            # one robust re-fit
        A = np.stack([np.ones_like(xs), xs, xs ** 2], 1) * ws[:, None]
        coeffs = np.linalg.lstsq(A, cs * ws, rcond=None)[0]
        res = np.abs(cs - (coeffs[0] + coeffs[1] * xs + coeffs[2] * xs ** 2))
        keep = res < max(0.5, 2.5 * np.median(res))
        if keep.all() or keep.sum() < 3:
            break
        xs, cs, ws = xs[keep], cs[keep], ws[keep]
    return dict(coeffs=coeffs.tolist(), x_min=float(xs.min()), x_max=float(xs.max()),
                n_bins=int(len(xs)))


class LanePerception:
    def __init__(self, onnx_path, camera=None, every=2):
        import os
        import onnxruntime as ort
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f'YOLOP weights not found: {onnx_path} '
                                    f'(B2D_YOLO=1 ./tools/b2d/setup_b2d.sh downloads them)')
        prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, providers=prov)
        self.cam = camera or CameraGeometry()
        self.every = every
        self.n = 0
        self.last = dict(valid=False)

    def masks(self, img_bgr):
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        canvas, r, dw, dh, nw, nh = letterbox(rgb)
        x = ((canvas / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        _, da, ll = self.sess.run(['det_out', 'drive_area_seg', 'lane_line_seg'],
                                  input_feed={'images': x})
        da = np.argmax(da[:, :, dh:dh + nh, dw:dw + nw], axis=1)[0]
        ll = np.argmax(ll[:, :, dh:dh + nh, dw:dw + nw], axis=1)[0]
        return da, ll, r

    def update(self, img_bgr):
        """Lane-centre estimate; YOLOP runs every `every` calls (the last
        estimate is returned in between)."""
        self.n += 1
        if (self.n - 1) % self.every:
            return self.last
        da, ll, r = self.masks(img_bgr)
        v, u = np.nonzero(ll)
        if len(u) < 20:
            self.last = dict(valid=False)
            return self.last
        X, Y = self.cam.pixels_to_ground(u / r, v / r)
        fit = fit_lane_centre(X, Y)
        self.last = dict(valid=fit is not None, **(fit or {}),
                         drivable_frac=float(da.mean()))
        return self.last

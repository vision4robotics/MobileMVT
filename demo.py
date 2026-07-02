from ultralytics import YOLO
import os, cv2
import argparse
import numpy as np

from tracker.uacmc import UCMCTrack, TrackStatus
from detector.mapper import Mapper


def xy_to_uv(mapper, x, y):
    A = mapper.A  # 3x3
    p = A @ np.array([x, y, 1.0], dtype=np.float32).reshape(3, 1)
    w = float(p[2, 0]) if abs(p[2, 0]) > 1e-9 else 1e-9
    u = float(p[0, 0] / w)
    v = float(p[1, 0] / w)
    return u, v
class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(2)  

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left + self.bb_width / 2, self.bb_top + self.bb_height, self.y[0, 0], self.y[1, 0])

    def __repr__(self):
        return self.__str__()

class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO(r'weights/vesselmot.pt')

    def get_dets(self, img, conf_thresh=0):
        dets = []
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(frame, imgsz=320, verbose=False)

        det_id = 0
        for box in results[0].boxes:
            conf = float(box.conf.cpu().numpy()[0])
            if conf <= conf_thresh:
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            x1, y1, x2, y2 = xyxy
            w = float(x2 - x1)
            h = float(y2 - y1)

            det = Detection(det_id)
            det.bb_left = float(x1)
            det.bb_top = float(y1)
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id

            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])

            dets.append(det)
            det_id += 1

        return dets


class PoseEstimator:
    def __init__(self, max_points=10, alpha=0.35,
                 ransac_thresh=3.0, max_iters=300, conf=0.99,
                 min_pts=3, max_angle_deg=8.0):
        self.max_points = int(max_points)
        self.alpha = float(alpha)
        self.ransac_thresh = float(ransac_thresh)
        self.max_iters = int(max_iters)
        self.conf = float(conf)
        self.min_pts = int(min_pts)
        self.max_angle = np.deg2rad(max_angle_deg)
        self._theta_prev = 0.0  

    @staticmethod
    def _centers_xy(boxes_xywh):
        return np.array([[x + 0.5 * w, y + h] for (x, y, w, h) in boxes_xywh], dtype=np.float32)

    def estimate_with_ids(self, prev_with_ids, curr_with_ids, top_conf_ids=None):
        shared = list(set(prev_with_ids.keys()) & set(curr_with_ids.keys()))
        if not shared:
            return np.eye(3, dtype=np.float64)

        if top_conf_ids is None:
            shared.sort()
        else:
            order = {tid: i for i, tid in enumerate(top_conf_ids)}
            shared.sort(key=lambda tid: order.get(tid, 1e9))
        if len(shared) > self.max_points:
            shared = shared[:self.max_points]

        prev_boxes = [prev_with_ids[tid][:4] for tid in shared]
        curr_boxes = [curr_with_ids[tid][:4] for tid in shared]
        if len(prev_boxes) < self.min_pts:
            return np.eye(3, dtype=np.float64)

        P = self._centers_xy(prev_boxes)
        Q = self._centers_xy(curr_boxes)

        M, inliers = cv2.estimateAffinePartial2D(
            P, Q, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters, confidence=self.conf
        )
        if M is None:
            return np.eye(3, dtype=np.float64)

        a, b = M[0, 0], M[1, 0]
        s = np.hypot(a, b) + 1e-12
        theta = np.arctan2(M[1, 0] / s, M[0, 0] / s)

        if abs(theta) > self.max_angle:
            theta_s = 0.0
        else:
            theta_s = self.alpha * theta + (1.0 - self.alpha) * self._theta_prev
        self._theta_prev = theta_s

        c, s = np.cos(theta_s), np.sin(theta_s)
        dR = np.eye(3, dtype=np.float64)
        dR[0, 0] = c; dR[0, 1] = -s
        dR[1, 0] = s; dR[1, 1] =  c
        return dR


def main(args):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not os.path.exists('output'):
        os.makedirs('output')
    video_out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, None)
    pose = PoseEstimator(max_points=10, alpha=0.35,
                         ransac_thresh=3.0, max_iters=300, conf=0.99,
                         min_pts=3, max_angle_deg=8.0)

    dR = np.eye(3, dtype=np.float64) 
    prev_with_ids = {}

    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:
            break
        if not np.allclose(dR, np.eye(3)):
            detector.mapper.apply_delta_pose(delta_R=dR, delta_T=np.zeros((3, 1)))
        dR = np.eye(3, dtype=np.float64)

        dets = detector.get_dets(frame_img, args.conf_thresh)
        tracker.update(dets, frame_id)
        for trk in tracker.trackers:
            pred_xy = getattr(trk, "pred_xy", None)
            if pred_xy is None:
                pred_xy = trk.kf.H @ trk.kf.x  

            px = float(pred_xy[0, 0])
            py = float(pred_xy[1, 0])

            u, v = xy_to_uv(detector.mapper, px, py)

            w = float(trk.w) if getattr(trk, "w", 0) > 0 else 40.0
            h = float(trk.h) if getattr(trk, "h", 0) > 0 else 20.0


            x1p = int(u - w / 2)
            y1p = int(v - h)
            x2p = int(u + w / 2)
            y2p = int(v)

            Hh, Ww = frame_img.shape[:2]
            x1p = max(0, min(Ww - 1, x1p)); y1p = max(0, min(Hh - 1, y1p))
            x2p = max(0, min(Ww - 1, x2p)); y2p = max(0, min(Hh - 1, y2p))

            cv2.rectangle(frame_img, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            cv2.putText(frame_img, str(trk.id), (x1p, y1p),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        curr_with_ids = {}
        for det in dets:
            if det.track_id > 0:
                curr_with_ids[det.track_id] = (det.bb_left, det.bb_top, det.bb_width, det.bb_height, det.conf)
        if prev_with_ids and curr_with_ids:
            ids_sorted = sorted(curr_with_ids.keys(), key=lambda tid: curr_with_ids[tid][4], reverse=True)
            dR = pose.estimate_with_ids(prev_with_ids, curr_with_ids, top_conf_ids=ids_sorted)
        prev_with_ids = curr_with_ids

        frame_id += 1

        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)
        video_out.write(frame_img)

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="demo/vesselmot-demo.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default="demo/vesselmot.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=4.20, help='wx')
parser.add_argument('--wy', type=float, default=2.60, help='wy')
parser.add_argument('--vmax', type=float, default=1.0, help='vmax')
parser.add_argument('--a', type=float, default=5.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=3.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.58, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.29, help='detection confidence threshold')
args = parser.parse_args()

main(args)


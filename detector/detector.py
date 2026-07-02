from .mapper import Mapper
from .gmc import GMCLoader
import numpy as np
import cv2
import csv

class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)
        

    def get_box(self):
        return [self.bb_left, self.bb_top, self.bb_width, self.bb_height]


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()

class Detector:
    def __init__(self, add_noise = False):
        self.seq_length = 0
        self.gmc = None
        self.add_noise = add_noise
        self.pose_delta = {}   # frame -> (theta_rad, tx_px, ty_px)

    def load(self,cam_para_file, det_file, pose_delta_file=None):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.pose_delta.clear()
        if pose_delta_file is not None:
            with open(pose_delta_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row: 
                        continue
                    frame = int(row["frame"])
                    theta = float(row["theta_rad"])
                    self.pose_delta[frame] = theta
        self.load_detfile(det_file)

    def load_detfile(self, filename):
        self.dets = {}
        self.prev_boxes_for_pose = None  
        self.seq_length = 0

        cur_frame_id = None
        frame_dets = []

        def apply_offline_rotation(frame_id):
            theta = self.pose_delta.get(frame_id, 0.0)
            if abs(theta) < 1e-6:
                return
            c, s = np.cos(theta), np.sin(theta)
            dR = np.eye(3, dtype=np.float64)
            dR[0, 0] = c; dR[0, 1] = -s
            dR[1, 0] = s; dR[1, 1] =  c
            self.mapper.apply_delta_pose(delta_R=dR, delta_T=np.zeros((3,1)))


        def process_one_frame(frame_id, dets_this_frame):
           
            apply_offline_rotation(frame_id)

            for det in dets_this_frame:
                if self.add_noise:
                    noise_z = (0.5 / 180.0 * np.pi) if (frame_id % 2 == 0) else (-0.5 / 180.0 * np.pi)
                    self.mapper.disturb_campara(noise_z)

                det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])

                if self.add_noise:
                    self.mapper.reset_campara()
            self.dets[frame_id] = dets_this_frame

        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                frame_id = int(line[0])
                if frame_id > self.seq_length:
                    self.seq_length = frame_id
                det_id = int(line[1])

                if cur_frame_id is not None and frame_id != cur_frame_id:
                    process_one_frame(cur_frame_id, frame_dets)
                    frame_dets = []
                det = Detection(det_id)
                det.bb_left = float(line[2])
                det.bb_top = float(line[3])
                det.bb_width = float(line[4])
                det.bb_height = float(line[5])
                det.conf = float(line[6])
                det.det_class = int(line[7])
                if det.det_class == -1:
                    det.det_class = 0
                frame_dets.append(det)
                cur_frame_id = frame_id
        if cur_frame_id is not None and len(frame_dets) > 0:
            process_one_frame(cur_frame_id, frame_dets)

   
            
    def get_dets(self, frame_id,conf_thresh = 0,det_class = 0):
        dets = self.dets[frame_id]
        dets = [det for det in dets if det.det_class == det_class and det.conf >= conf_thresh]
        return dets
    
    
    def cmc(self,x,y,w,h,frame_id):
        u,v = self.mapper.xy2uv(x,y)
        affine = self.gmc.get_affine(frame_id)
        M = affine[:,:2]
        T = np.zeros((2,1))
        T[0,0] = affine[0,2]
        T[1,0] = affine[1,2]

        p_center = np.array([[u],[v-h/2]])
        p_wh = np.array([[w],[h]])
        p_center = np.dot(M,p_center) + T
        p_wh = np.dot(M,p_wh)

        u = p_center[0,0]
        v = p_center[1,0]+p_wh[1,0]/2

        xy,_ = self.mapper.uv2xy(np.array([[u],[v]]),np.eye(2))

        return xy[0,0],xy[1,0]



from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple
from multiprocessing.connection import Client

import cv2
import face_recognition

import os
import math
import queue
import numpy as np
import threading
import time
import concurrent.futures

face_count = 0
discard_count = 0
address = ('localhost', 6000)
# address = None

@dataclass
class tracking_data:
    pos: Tuple[int, int]
    update: float
    score: float

class face_score:
    @dataclass
    class Data:
        id: int
        enc: np.ndarray
        img: np.ndarray
        size: tuple
        scores: dict

    def __init__(self, address=None):
        if address is not None:
            self.address = address
            self.conn = Client(address)
        self.debug = self.config['Debug']['Verbose_Logs']
        self.tracking = []
        self._frame = 30
        self.init_time = time.time()
        self.t = 0

    config = {
        'PreFilter': {
            'Size': {
                'Enabled': True,
                'Discard_TH': 90,
            },
            'Canny': {
                'Enabled': True,
                'Discard_TH': 5,
                'P1': 50,
                'P2': 1500,
                'Aperture': 5,
                'L2gradient': True,
                'Debug': False
            },
        },
        'Criteria': {
            'Shape': {
                'Enabled': True,
                'Alignment_Penalty': 0.8
            },
            'Pose': {
                'Enabled': True,
                'Alignment_Penalty_1': 0.5,
                'Alignment_Penalty_2': 0.3
            },
            'Size': {
                'Enabled': True,
                'Thresholds': [120, 180, 240, 400, 600],
                'Multiply': [0.5, 0.8, 1, 1.05, 1.1, 1.15],
            },
            'Canny': {
                'Enabled': True,
                'Thresholds': [10, 15, 20, 25],
                'Multiply': [0.5, 0.85, 1, 1.1, 1.15],
            },
            'Average_Shape_Pose_Min_Mult': 0.7,
            'Average_Shape_Pose_Max_Mult': 0.3
        },
        'PostFilter': {
            'Tracking': {
                'Enabled': True,
                'Timeout': 10,
                'Distance': 80,
            },
            'Shape_Discard_TH': 70,
            'Pose_Discard_TH': 55,
        },
        'Settings': {
            'Capture_Resize': None,  # (w, h)
            'Output_Scale': 1.5,
            'Skip_Frames': 30,
            'Num_CPU': 1,
            'Input_Path': 'X:/eor/backend/recognition/tests/samples/modern 720.mp4',
            'Starting_Frame': 300,
        },
        'Debug': {
            'Draw_Face_Rect': True,
            'Draw_Landmarks': True,
            'Draw_Scores': True,
            'Draw_Id': True,
            'Verbose_Logs': False,
            'No_Discard' : False,
            'Save_Frame' : False,
            'Save_Discard' : False,
            'Save_Face' : False,
            'Debug_Files_Path': 'R:/Faces/',
        }
    }

    def run(self, frame, timestamp):
        results = []
        self.t = time.time()

        # Detect faces
        face_locs = face_recognition.face_locations(frame) # (top, right, bottom, left) order
        encodes, raw_landmarks = face_recognition.face_encodings_landmarks(frame, face_locs, model='large')

        for i, encode in enumerate(encodes):
            face_loc = face_locs[i]
            landmarks = self.process_landmarks(raw_landmarks[i])
            face, face_h, face_w = self.crop_face(frame, face_locs[i])
            out_face, out_h, out_w = self.crop_face(frame, face_locs[i], scale=True)

            scores = { 'discard': [], 'final': -1, 'shape': -1, 'pose': -1, 'size': -1 }
            scores['size_raw'] = (face_w, face_h)

            discard = self.pre_filter(face, scores)
            self.criteria(landmarks, face, scores)
            discard |= self.post_filter(face_loc, scores, timestamp)

            if discard:
                global discard_count
                if self.config['Debug']['Save_Discard']:
                    self.draw_debug(frame, face_loc, landmarks, scores, face_count)
                    try:
                        out_face = cv2.copyMakeBorder(out_face, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                        cv2.imwrite(os.path.join(self.config['Debug']['Debug_Files_Path'] , f"discard_{'-'.join(scores['discard'])}_{timestamp:.2f}s_#{discard_count}.jpg"), out_face)
                        print(f'-{discard_count} {timestamp:.2f} discard face:\t{scores}')
                    except Exception as e:
                        print(e)

                if self.config['Debug']['No_Discard']:
                    out_face = self.color_filter(out_face)

                discard_count += 1
                continue

            self.draw_debug(frame, face_loc, landmarks, scores, face_count)
            if len(scores['discard']) == 0:
                self.tracking.append(tracking_data(face_loc, timestamp, scores['final']))

            print(f'#{face_count} {timestamp:.2f} \t{scores}')
            results.append(self.Data(0, encode, out_face, (out_w, out_h), scores))

            if self.config['Settings']['Num_CPU'] > 1:
                print(f'p: {os.getpid()} t: ({(time.time() - self.t):.2f}/{(time.time() - self.init_time):.2f}s) appnd face: {scores}')


        if self.config['Debug']['Save_Frame']:
            cv2.imwrite(os.path.join(self.config['Debug']['Debug_Files_Path'] , f'{id}f.jpg'), frame)

        self.cleanup(timestamp)
        return results

    def pre_filter(self, face, scores):
        discard = False
        if self.config['PreFilter']['Canny']['Enabled']:
            scores['canny_raw'] = self.canny_mean(face)
            if scores['canny_raw'] < self.config['PreFilter']['Canny']['Discard_TH']:
                scores['discard'].append('canny')
                discard = True
        if self.config['PreFilter']['Size']['Enabled']:
            if (scores['size_raw'][0] + scores['size_raw'][0]) / 2 < self.config['PreFilter']['Size']['Discard_TH']:
                scores['discard'].append('size')
                discard = True
        return discard

    def criteria(self, landmarks, face, scores):
        if self.config['Criteria']['Shape']['Enabled']:
            self.shape_score(landmarks, scores['size_raw'][1], scores['size_raw'][0], scores)
        if self.config['Criteria']['Pose']['Enabled']:
            self.pose_score(landmarks, scores['size_raw'][1], scores['size_raw'][0], scores)
        if self.config['Criteria']['Size']['Enabled']:
            self.size_score(scores['size_raw'][1], scores['size_raw'][0], scores)
        if self.config['Criteria']['Canny']['Enabled']:
            self.canny_score(scores['canny_raw'], scores)
        self.combine_scores(scores)

    def canny_score(self, canny_raw, scores):
        scores['canny'] = round(self.linearize(self.config['Criteria']['Canny']['Thresholds'], self.config['Criteria']['Canny']['Multiply'], canny_raw), 2)

    # ths: LOW  0|  1|  2|  3|  4|  5|   HIGH
    # mls:     0x  1x  2x  3x  4x  5x  6x
    def linearize(self, thresholds, multipliers, input):
        m = 0
        ix = 0
        for i, th in enumerate(thresholds):
            if input < th:
                m = multipliers[i]
                ix = i
                break
        m = multipliers[-1] if m == 0 else m
        ix = len(thresholds) - 1 if ix == 0 else ix

        if ix == 0 or ix == len(thresholds) - 1:
            return m

        dh = abs(input - thresholds[ix + 1])
        dl = abs(input - thresholds[ix - 1])
        dt = thresholds[ix + 1] - thresholds[ix - 1]

        return (dh / dt) * multipliers[ix + 1] + (dl / dt) * multipliers[ix - 1]

    def post_filter(self, face_loc, scores, timestamp):
        discard = False
        if self.config['PostFilter']['Tracking']['Enabled']:
            discard = self.face_tracking(face_loc, scores, timestamp)

        if scores['shape'] < self.config['PostFilter']['Shape_Discard_TH']:
            scores['discard'].append('shape')
            discard = True

        if scores['pose'] < self.config['PostFilter']['Pose_Discard_TH']:
            scores['discard'].append('pose')
            discard = True

        return discard

    def color_filter(self, img):
        return cv2.bitwise_not(img)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config['Settings']['Capture_Resize']:
            frame = cv2.resize(frame, self.config['Settings']['Capture_Resize'])
        return frame

    def cleanup(self, timestamp):
        self.tracking = list(filter(lambda t: timestamp - t.update < self.config['PostFilter']['Tracking']['Timeout'], self.tracking))

    def face_tracking(self, face_loc, scores, timestamp):
        discard = False
        for t in self.tracking:
            if math.dist(t.pos, face_loc) < self.config['PostFilter']['Tracking']['Distance']:
                if t.score >= scores['final']:
                    discard = True
                else:
                    t.update = timestamp
                    t.score = scores['final']
                break

        if discard:
            scores['discard'].append('tracking')

        return discard

    def process_landmarks(self, raw_landmarks: list) -> list:
        """
                .10     .11            .12      .13

                 (  .7 )                (  .9  )
                                |
                                .8
            .5                  |                   .6

                                .3
                               MOUTH
                                .4
                .1                              .2

                                .0
        """
        landmarks = []
        for landmark_group in [
                [8],             # 0: Chin
                range(3, 6),     # 1: bottom left  cheek
                range(11, 14),   # 2: bottom right cheek
                range(49, 55),   # 3: upper lip
                range(55, 60),   # 4: bottom lip
                range(1, 3),     # 5: upper left cheek
                range(14, 17),   # 6: upper right cheek
                range(36, 42),   # 7: left eye
                range(28, 31),   # 8: nose
                range(42, 48),   # 9: right eye
                range(17, 19),   # 10: left of left eyebrow
                range(20, 22),   # 11: right of left eyebrow
                range(22, 24),   # 12: left of right eyebrow
                range(25, 27)]:  # 13: right of right eyebrow
            xs = list(map(lambda n: raw_landmarks.part(n).x, landmark_group))
            ys = list(map(lambda n: raw_landmarks.part(n).y, landmark_group))
            landmarks.append((int(sum(xs) / len(xs)), int(sum(ys) / len(ys))))
        return landmarks

    def crop_face(self, frame: np.ndarray, face_loc, scale=False):
        crop_h = abs(face_loc[3] - face_loc[1])
        crop_w = abs(face_loc[0] - face_loc[2])
        crop_h_ext = 0
        crop_w_ext = 0
        scale_mult = 1
        if scale:
            scale_mult = self.config['Settings']['Output_Scale']
            crop_h_ext = int(((scale_mult - 1) * crop_h) / 2)
            crop_w_ext = int(((scale_mult - 1) * crop_w) / 2)
        return frame[face_loc[0] - crop_h_ext : face_loc[0] + crop_h + crop_h_ext,
               face_loc[3] - crop_w_ext : face_loc[3] + crop_w + crop_w_ext], crop_h * scale_mult, crop_w * scale_mult

    def shape_score(self, landmarks, face_h, face_w, scores):
        lx = [landmarks[i][0] for i, l in enumerate(landmarks)]
        ly = [landmarks[i][1] for i, l in enumerate(landmarks)]
        calc = (lambda a, b: (face_h - abs(a - b)) / face_h)
        calc_x = (lambda a, b: calc(lx[a], lx[b]))
        calc_y = (lambda a, b: calc(ly[a], ly[b]))
        shape_score = []
        shape_score.append(calc_y(1, 2))
        shape_score.append(calc_y(5, 6))
        shape_score.append(calc_y(7, 9))
        shape_score.append(calc_y(1, 2))
        shape_score.append(calc_y(10, 13))
        shape_score.append(calc_y(11, 12))
        shape_score.append(calc_x(0, 8))
        shape_score.append(calc_x(3, 4))
        if self.debug:
            scores['shape_raw'] = [round(x, 2) for x in shape_score]

        if (self.config['Criteria']['Shape']['Alignment_Penalty'] != 0 and
                (ly[10] > ly[11] > ly[12] > ly[13] or ly[10] < ly[11] < ly[12] < ly[13])):
            shape_score = list(
                map(lambda s: s * self.config['Criteria']['Shape']['Alignment_Penalty'], shape_score))
        scores['shape'] = round(sum(shape_score) / len(shape_score), 2)

    def pose_score(self, landmarks, face_h, face_w, scores):
        lx = [landmarks[i][0] for i, l in enumerate(landmarks)]

        # p1 - p2 == p3 - p4
        def calc(p1, p2, p3, p4):
            d1 = lx[p1] - lx[p2]
            d2 = lx[p3] - lx[p4]
            mn = min(d1, d2)
            mx = max(d1, d2)
            mm = 1 - (mx - mn) / mx
            wmn = 1 - abs(face_w / 2 - d1) / max(face_w / 2, d1)
            wmx = 1 - abs(face_w / 2 - d2) / max(face_w / 2, d2)
            return (((wmn + wmx) / 2) + mm) / 2

        pose_score = []
        pose_score.append(calc(8, 5, 6, 8))
        pose_score.append(calc(7, 5, 6, 9))
        pose_score.append(calc(3, 1, 2, 3))
        pose_score.append(calc(4, 1, 2, 4))
        pose_score.append(calc(11, 10, 13, 12))
        if self.debug:
            scores['pose_raw'] = [round(x, 2) for x in pose_score]

        if not (lx[6] > lx[9] > lx[8] > lx[7] > lx[5]):
            pose_score[0] *= self.config['Criteria']['Pose']['Alignment_Penalty_1']
            pose_score[1] *= self.config['Criteria']['Pose']['Alignment_Penalty_1']
        if not ((lx[2] > lx[3]) or
                not (lx[2] > lx[4]) or
                not (lx[3] > lx[1]) or
                not (lx[4] > lx[1])):
            pose_score[2] *= self.config['Criteria']['Pose']['Alignment_Penalty_2']
            pose_score[3] *= self.config['Criteria']['Pose']['Alignment_Penalty_2']
        scores['pose'] = round(sum(pose_score) / len(pose_score), 2)

    def canny_mean(self, face_img):
        canny = cv2.Canny(face_img,
                          self.config['PreFilter']['Canny']['P1'],
                          self.config['PreFilter']['Canny']['P2'],
                          apertureSize=self.config['PreFilter']['Canny']['Aperture'],
                          L2gradient=self.config['PreFilter']['Canny']['L2gradient'])
        if self.config['PreFilter']['Canny']['Debug']:
            cv2.imwrite(os.path.join(self.config['Debug']['Debug_Files_Path'] , f'{self.id}c.jpg'), canny)
        mean = np.mean(canny)
        return round(mean, 2)

    def size_score(self, face_h, face_w, scores):
        l = (face_h + face_w) / 2
        scores['size'] = round(self.linearize(self.config['Criteria']['Size']['Thresholds'], self.config['Criteria']['Size']['Multiply'], l), 2)

    def combine_scores(self, scores):
        shape_pose = 0
        canny = 0
        size = 0
        final = 0
        # scores = {k:v * 100 for k, v in scores.items() if v != None}
        if 'shape' in scores and 'pose' in scores:
            scores['shape'] *= 100
            scores['pose'] *= 100
            shape_pose = (max(scores['shape'], scores['pose']) * self.config['Criteria']['Average_Shape_Pose_Max_Mult'] +
                          min(scores['shape'], scores['pose']) * self.config['Criteria']['Average_Shape_Pose_Min_Mult'])
        elif 'shape' in scores:
            scores['shape'] *= 100
            shape_pose = scores['shape']
        elif 'pose' in scores:
            scores['pose'] *= 100
            shape_pose = scores['pose']

        if 'canny' in scores.keys():
            scores['canny'] *= 100
            scores['canny'] = round(scores['canny'], 2)
            canny = scores['canny']

        l = [shape_pose, shape_pose, canny]
        final = round(sum(l) / len(l))

        if 'size' in scores.keys():
            size = scores['size']
            final *= size

        scores['final'] = final

    def draw_debug(self, frame, face_loc, landmark, scores, id):
        put_text = (lambda str, pos: cv2.putText(frame, str, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA))
        if self.config['Debug']['Draw_Face_Rect']:
            cv2.rectangle(frame, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 255, 255), 1)

        if self.config['Debug']['Draw_Landmarks']:
            for i, point in enumerate(landmark):
                cv2.circle(frame, point, 1, (0, 0, 0), -1)
                # put_text(f"{i}", point)

        if self.config['Debug']['Draw_Scores']:
            crop_h = abs(face_loc[3] - face_loc[1])
            crop_h_ext = ((self.config['Settings']['Output_Scale'] - 1) * crop_h) / 2
            put_text(f"f{int(scores['final'])}", (face_loc[3], face_loc[0]))
            put_text(f"s{int(scores['shape'])}", (face_loc[1] - 20, face_loc[0]))
            put_text(f"p{int(scores['pose'])}", (face_loc[3], face_loc[2]))
            put_text(f"i{int(scores['size'] * 100)}", (face_loc[1] - 20, face_loc[2]))
            # put_text(f"c{int(scores['canny'])}", (face_loc[3] - 20, face_loc[2] + int(crop_h_ext / 2)))

        if self.config['Debug']['Draw_Id']:
            put_text(f"#{id}", (face_loc[1] - 20,
                     face_loc[2] + int(crop_h_ext / 2)))

    def send(self, data):
        msg = asdict(data)
        try:
            self.conn.send(msg)
        except Exception as e:
            return False
        return True

    def __del__(self):
        if self.conn:
            self.conn.close()

q = queue.Queue()
def sender_loop(fd: face_score):
    while True:
        data, timestamp = q.get()
        global face_count
        data.id = face_count

        if fd.config['Debug']['Save_Face']:
            cv2.imwrite(os.path.join(fd.config['Debug']['Debug_Files_Path'] , f'good_{timestamp:.2f}s_#{face_count}.jpg'), data.img)
            # print(f'#{data.id} ({(time.time() - fd.init_time):5.2f}s) save\tface: {data.scores}')

        if address is not None:
            fd.send(data)

        face_count += 1
        q.task_done()

def main():
    start = time.perf_counter()
    fd = face_score(address)
    cap = cv2.VideoCapture(fd.config['Settings']['Input_Path'])
    frame_num = fd.config['Settings']['Starting_Frame']

    threading.Thread(target=sender_loop, args=(fd,)).start()
    while (fd.config['Settings']['Num_CPU'] == 1):
        frame_num += fd.config['Settings']['Skip_Frames']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            res = fd.run(frame, cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            if res is not None:
                for r in res:
                    if r is not None:
                        q.put((r, cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
        else:
            # print(f'video capture error! {ret}')
            return

    while True:
        frames = []
        for _ in range(fd.config['Settings']['Num_CPU']):
            frame_num += fd.config['Settings']['Skip_Frames']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # print(f'video capture error! {ret}')
                return

        with concurrent.futures.ProcessPoolExecutor(max_workers=fd.config['Settings']['Num_CPU']) as executor:
            results = [executor.submit(fd.run, frame, cap.get(cv2.CAP_PROP_POS_MSEC)/1000) for frame in frames]

            for f in concurrent.futures.as_completed(results):
                if f.result is None:
                    continue
                for elem in f.result():
                    if elem is not None:
                        q.put((elem, cap.get(cv2.CAP_PROP_POS_MSEC)/1000))


if __name__ == '__main__':
    main()
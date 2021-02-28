"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

from KalmanFilter1D import Kalman1D
import cv2
from gaze_tracking import GazeTracking
from scipy.interpolate import interp1d
import numpy as np

gaze = GazeTracking()
webcam = cv2.VideoCapture("v4l2src device=/dev/video0 ! videorate ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=2 drop=true", cv2.CAP_GSTREAMER)

H_MIN = 0.5
H_MAX = 1.0
V_MIN = 0.4
V_MAX = 1.0

kf = Kalman1D(R=0.01**2)

class GazeTracker(object):

    def __init__(self, R_var=0.01**2) -> None:
        super().__init__()
        self.kf = Kalman1D(R_var)
        self.h_ratio = (H_MAX+H_MIN)/2
        self.v_ratio = (V_MAX + V_MIN)/2
        
        self.f_height = 480
        self.f_width = 640

        self.meas_x = int(self.f_width/2)
        self.meas_y = int(self.f_height/2)

        self.linear_map_v = interp1d([V_MIN,V_MAX],[0, self.f_height])
        self.linear_map_h = interp1d([H_MIN, H_MAX],[0, self.f_width])

    def get_measurement(self, h_ratio, v_ratio):
        # Clip ratios within acceptable range
        h_ratio = max(min(h_ratio, H_MAX), H_MIN)
        v_ratio = max(min(v_ratio, V_MAX), V_MIN)

        self.meas_x = int(self.linear_map_h(h_ratio))
        self.meas_y = int(self.linear_map_v(v_ratio))

        return self.meas_x, self.meas_y

    def get_estimate(self, meas_x, meas_y):
        est_gaze = self.kf.update(meas_x + 1j*meas_y)
        return int(np.real(est_gaze)), int(np.imag(est_gaze))

    def get_estimate_missing_meas(self):
        meas_x = self.meas_x
        meas_y = self.meas_y

        cur_meas_var = kf.R
        kf.update_meas_variance(new_meas_var=100)
        est_gaze_x, est_gaze_y = tracker.get_estimate(meas_x, meas_y)
        kf.update_meas_variance(new_meas_var=cur_meas_var)

        return est_gaze_x, est_gaze_y

    def update_eyes(self, frame, eye_left, eye_right, show=True):
        # Get left eye width, height, and bounding box top left
        eye_l_dim = int(eye_left.center[0]*2), int(eye_left.center[1]*2)
        eye_l_origin = int(eye_left.origin[0]), int(eye_left.origin[1])

        # Get right eye width, height, and bounding box top left
        eye_r_dim = int(eye_right.center[0]*2), int(eye_right.center[1]*2)
        eye_r_origin = int(eye_right.origin[0]), int(eye_right.origin[1])

        # Update center coordinate
        self.update_face_center(eye_l_origin, eye_l_dim, eye_r_origin)

        if show:
            self.show_eyes(frame, eye_l_origin, eye_l_dim, eye_r_origin, eye_r_dim)
            

    def show_eyes(self, frame, eye_l_origin, eye_l_dim, eye_r_origin, eye_r_dim):
        eye_l_ox, eye_l_oy = eye_l_origin
        eye_r_ox, eye_r_oy = eye_r_origin

        eye_l_w, eye_l_h = eye_l_dim
        eye_r_w, eye_r_h = eye_r_dim

        cv2.rectangle(frame, (eye_l_ox, eye_l_oy), (eye_l_ox+eye_l_w, eye_l_oy+eye_l_h), (255,0,0), 2)
        cv2.rectangle(frame, (eye_r_ox, eye_r_oy), (eye_r_ox+eye_r_w, eye_r_oy+eye_r_h), (255,0,0), 2)

    def update_face_center(self, eye_l_origin, eye_l_dim, eye_r_origin):
        eye_l_ox, eye_l_oy = eye_l_origin
        eye_r_ox, _ = eye_r_origin
        eye_l_w, eye_l_h = eye_l_dim

        self.center_coord_x = int(((eye_l_ox + eye_l_w) + eye_r_ox)/2)
        self.center_coord_y = int(((eye_l_oy + eye_l_h) + eye_l_oy)/2)

    def show_gaze_vector(self, frame, est_gaze_x, est_gaze_y, use_eye_center=True):
        if use_eye_center:
            arrow_start = (self.center_coord_x, self.center_coord_y)
        else:
            arrow_start = (int(self.f_width/2), int(self.f_height/2))
            
        cv2.arrowedLine(frame, arrow_start , (est_gaze_x, est_gaze_y), (0, 255, 0), 9)



tracker = GazeTracker()

while True:
    # We get a new frame from the webcam
    ret, frame = webcam.read()

    if ret:
        frame = cv2.flip(frame, 1)
        # We send this frame to GazeTracking to analyze it
        try:
            gaze.refresh(frame)
        except:
            continue

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        h_ratio = gaze.horizontal_ratio()
        v_ratio = gaze.vertical_ratio()
        # print('Ratios: ', h_ratio, v_ratio)


        if h_ratio and v_ratio:            
            # print('h_ratio: {}, v_ratio: {}'.format(h_ratio, v_ratio))
            meas_gaze_x, meas_gaze_y = tracker.get_measurement(h_ratio, v_ratio)
            est_gaze_x, est_gaze_y = tracker.get_estimate(meas_gaze_x, meas_gaze_y)
            # print('est_gaze_x: {}, est_gaze_y: {}'.format(est_gaze_x, est_gaze_y))
            print('diff_x: {}, diff_y: {}'.format(meas_gaze_x - est_gaze_x, meas_gaze_x - est_gaze_y))

            tracker.update_eyes(frame, gaze.eye_left, gaze.eye_right)
            tracker.show_gaze_vector(frame, est_gaze_x, est_gaze_y)
        else:
            est_gaze_x, est_gaze_y = tracker.get_estimate_missing_meas()
            tracker.show_gaze_vector(frame, est_gaze_x, est_gaze_y, use_eye_center=False)
            
        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(frame, "H {} V {}: ".format(str(gaze.horizontal_ratio()), str(gaze.vertical_ratio())), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

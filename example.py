"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from scipy.interpolate import interp1d

gaze = GazeTracking()
webcam = cv2.VideoCapture("v4l2src device=/dev/video0 ! videorate ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=2 drop=true", cv2.CAP_GSTREAMER)

H_MIN = 0.5
H_MAX = 1.0
V_MIN = 0.4
V_MAX = 1.0

while True:
    # We get a new frame from the webcam
    ret, frame = webcam.read()

    if ret:
        frame = cv2.flip(frame, 1)
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

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
        print('Ratios: ', h_ratio, v_ratio)


        if h_ratio and v_ratio:
            h_ratio = max(min(h_ratio, H_MAX), H_MIN)
            v_ratio = max(min(v_ratio, V_MAX), V_MIN)
            # v_ratio = 1 - min(v_ratio, 1)
            f_height, f_width = frame.shape[0], frame.shape[1]
            center_coords = int(f_width/2), int(f_height/2)
            
            linear_map_v = interp1d([V_MIN,V_MAX],[0, f_height])
            linear_map_h = interp1d([H_MIN, H_MAX],[0, f_width])

            print('h_ratio: {}, v_ratio: {}'.format(h_ratio, v_ratio))
            arrow_end_x, arrow_end_y = int(linear_map_h(h_ratio)), int(linear_map_v(v_ratio))
            print('arrow_x: {}, arrow_y: {}'.format(arrow_end_x, arrow_end_y))
            cv2.arrowedLine(frame, center_coords, (arrow_end_x, arrow_end_y), (0, 255, 0), 9)

            # Get left eye width, height, and bounding box top left
            eye_l_w, eye_l_h = int(gaze.eye_left.center[0]*2), int(gaze.eye_left.center[1]*2)
            eye_l_ox, eye_l_oy = int(gaze.eye_left.origin[0]), int(gaze.eye_left.origin[1])
            cv2.rectangle(frame, (eye_l_ox, eye_l_oy), (eye_l_ox+eye_l_w, eye_l_oy+eye_l_h), (255,0,0), 2)

            # Get right eye width, height, and bounding box top left
            eye_r_w, eye_r_h = int(gaze.eye_right.center[0]*2), int(gaze.eye_right.center[1]*2)
            eye_r_ox, eye_r_oy = int(gaze.eye_right.origin[0]), int(gaze.eye_right.origin[1])
            cv2.rectangle(frame, (eye_r_ox, eye_r_oy), (eye_r_ox+eye_r_w, eye_r_oy+eye_r_h), (255,0,0), 2)

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "H {} V {}: ".format(str(gaze.horizontal_ratio()), str(gaze.vertical_ratio())), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

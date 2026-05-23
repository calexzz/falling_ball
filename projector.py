import cv2
import numpy as np

import zmq
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://84.237.21.36:6002")

count = 0
while True:
    msg = socket.recv()
    print(len(msg))
    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)
    cv2.putText(frame, f"Count{count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    cv2.imshow("Stream", frame)
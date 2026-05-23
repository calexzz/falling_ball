import cv2
import numpy as np
from screeninfo import get_monitors

def show_on_monitor(mask):
    monitors = get_monitors
    if len(monitors < 2):
        print("Обнаружен лишь один монитор")
        target_monitor = monitors[0]
    else:
        target_monitor = monitors[1]

    cv2.namedWindow("Mask", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("Mask", target_monitor.x, target_monitor.y)
    cv2.setWindowProperty("Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Mask", mask)

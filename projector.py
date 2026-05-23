import cv2
import numpy as np

# pip install screeninfo
from screeninfo import get_monitors

# Передаём наше изображение 
def show_on_monitor(mask):
    # Получаем количество и информацию о мониторах
    monitors = get_monitors
    # Считаем, есть ли второй монитор
    if len(monitors < 2):
        print("Обнаружен лишь один монитор")
        target_monitor = monitors[0]
    else:
        target_monitor = monitors[1]

    # Выводим на проектор полноэкранное изображение
    cv2.namedWindow("Mask", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("Mask", target_monitor.x, target_monitor.y)
    cv2.setWindowProperty("Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Mask", mask)
# Нужно вставлять при выводе, после всех операций над изображением

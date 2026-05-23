import cv2
import numpy as np

CAMERA_INDEX = 0

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 80
HOUGH_MIN_LENGTH = 80
HOUGH_MAX_GAP = 20

BALL_MIN_AREA = 100
BALL_COLOR_BGR = (0, 0, 255)  # цвет шарика в BGR — красный
BALL_HUE_TOLERANCE = 15       # допуск по оттенку (±градусов)
BALL_SAT_MIN = 80             # минимальная насыщенность (0–255)
BALL_VAL_MIN = 80             # минимальная яркость (0–255)


def open_camera(index=CAMERA_INDEX):
    """Открывает захват с камеры."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def release_camera(cap):
    """Освобождает ресурсы камеры."""
    cap.release()


def read_frame(cap):
    """Считывает один кадр с камеры."""
    ret, frame = cap.read()
    return frame if ret else None


def bgr_to_hsv_hue(bgr):
    """Конвертирует BGR-цвет в оттенок HSV (0–179)."""
    pixel = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    return int(hsv[0, 0, 0])


def build_color_mask(hsv_frame, hue, hue_tolerance=BALL_HUE_TOLERANCE,
                     sat_min=BALL_SAT_MIN, val_min=BALL_VAL_MIN):
    """Строит бинарную маску для заданного оттенка в HSV-кадре."""
    lo_h = hue - hue_tolerance
    hi_h = hue + hue_tolerance

    if lo_h < 0:
        mask1 = cv2.inRange(hsv_frame,
                            np.array([lo_h + 180, sat_min, val_min]),
                            np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv_frame,
                            np.array([0, sat_min, val_min]),
                            np.array([hi_h, 255, 255]))
        return cv2.bitwise_or(mask1, mask2)

    if hi_h > 179:
        mask1 = cv2.inRange(hsv_frame,
                            np.array([lo_h, sat_min, val_min]),
                            np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv_frame,
                            np.array([0, sat_min, val_min]),
                            np.array([hi_h - 180, 255, 255]))
        return cv2.bitwise_or(mask1, mask2)

    return cv2.inRange(hsv_frame,
                       np.array([lo_h, sat_min, val_min]),
                       np.array([hi_h, 255, 255]))


def find_ball(frame, color_bgr=BALL_COLOR_BGR):
    """Находит цветной шарик на кадре с камеры."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = bgr_to_hsv_hue(color_bgr)
    mask = build_color_mask(hsv, hue)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < BALL_MIN_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def detect_surfaces(frame):
    """Находит прямолинейные отрезки на кадре — нарисованные линии."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if lines is None:
        return []

    return [tuple(line[0]) for line in lines]

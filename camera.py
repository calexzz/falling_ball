import cv2
import numpy as np

CAMERA_INDEX = 0

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 80
HOUGH_MIN_LENGTH = 80
HOUGH_MAX_GAP = 20

BALL_MIN_AREA = 100
BALL_BRIGHTNESS = 200


def open_camera(index=CAMERA_INDEX):
    """
    Открывает захват с камеры по заданному индексу устройства.
    Параметры:
        index (int): индекс USB-камеры (0 — первая доступная).
    Возвращает:
        cv2.VideoCapture: объект захвата.
    Исключения:
        RuntimeError: если камеру не удалось открыть.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def release_camera(cap):
    """
    Освобождает ресурсы камеры.
    Параметры:
        cap (cv2.VideoCapture): объект захвата, полученный из open_camera.
    """
    cap.release()


def read_frame(cap):
    """
    Считывает один кадр с камеры.
    Параметры:
        cap (cv2.VideoCapture): объект захвата.
    Возвращает:
        numpy.ndarray | None: BGR-кадр, или None если кадр не получен.
    """
    ret, frame = cap.read()
    return frame if ret else None


def detect_surfaces(frame):
    """
    Находит прямолинейные отрезки на кадре — нарисованные на доске линии.
    Алгоритм: размытие → Canny → вероятностное преобразование Хафа.
    Параметры:
        frame (numpy.ndarray): BGR-кадр с камеры.
    Возвращает:
        list[tuple[int, int, int, int]]: список отрезков вида (x1, y1, x2, y2)
        в пикселях камеры. Пустой список если линии не найдены.
    """
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


def find_ball(frame):
    """
    Находит белый проецируемый шарик на кадре с камеры.
    Алгоритм: пороговая бинаризация по яркости → эрозия/дилатация →
    поиск контуров → выбор наибольшего → центр масс.
    Параметры:
        frame (numpy.ndarray): BGR-кадр с камеры.
    Возвращает:
        tuple[int, int] | None: координаты центра шарика (cx, cy)
        в пикселях камеры, или None если шарик не обнаружен.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, BALL_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


if __name__ == "__main__":
    cap = open_camera()

    while True:
        frame = read_frame(cap)
        if frame is None:
            continue

        surfaces = detect_surfaces(frame)
        ball_pos = find_ball(frame)

        for x1, y1, x2, y2 in surfaces:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ball_pos:
            cv2.circle(frame, ball_pos, 10, (0, 0, 255), -1)
            print(f"Ball: {ball_pos}")

        cv2.imshow("camera", frame)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            break

    release_camera(cap)
    cv2.destroyAllWindows()

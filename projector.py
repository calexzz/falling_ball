import math
import pygame


# Цвета
BALL_COLOR      = (220, 90, 50)     # оранжево-красный шарик
BALL_OUTLINE    = (255, 160, 100)   # обводка шарика
BALL_MARKER     = (50, 20, 10)      # маркер вращения
PLATFORM_COLOR  = (80, 140, 80)     # зелёные платформы
PLATFORM_JOINT  = (120, 200, 120)   # соединения платформ
BG_COLOR        = (0, 0, 0)         # фон


def draw_ball(surface, state):
    """Рисует шарик на поверхности pygame.

    Параметры:
        surface: pygame.Surface — куда рисовать
        state: BallState — состояние шарика из physics.py
    """
    cx, cy = int(state.x), int(state.y)
    r = state.radius

    # Основной круг
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), r)
    # Обводка
    pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), r, 2)
    # Маркер вращения
    mx = cx + int((r - 4) * math.cos(state.angle))
    my = cy + int((r - 4) * math.sin(state.angle))
    pygame.draw.circle(surface, BALL_MARKER, (mx, my), 4)


def draw_platforms(surface, contours):
    """Рисует платформы на поверхности pygame.

    Параметры:
        surface: pygame.Surface — куда рисовать
        contours: list[((x1,y1), (x2,y2)), ...] — отрезки платформ
    """
    for (x1, y1), (x2, y2) in contours:
        pygame.draw.line(surface, PLATFORM_COLOR, (x1, y1), (x2, y2), 8)
        pygame.draw.circle(surface, PLATFORM_JOINT, (x1, y1), 4)
        pygame.draw.circle(surface, PLATFORM_JOINT, (x2, y2), 4)


def draw_camera_preview(surface, frame, width, height):
    """Рисует превью с камеры в углу экрана.

    Параметры:
        surface: pygame.Surface — куда рисовать
        frame: numpy.ndarray — кадр с камеры (BGR)
        width, height: int — размеры превью
    """
    if frame is None:
        return

    # BGR -> RGB, ресайз
    import cv2
    import numpy as np

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_small = cv2.resize(frame_rgb, (width, height))

    # numpy array -> pygame Surface
    frame_surface = pygame.surfarray.make_surface(frame_small.swapaxes(0, 1))
    surface.blit(frame_surface, (10, 10))

    # Рамка
    pygame.draw.rect(surface, (255, 255, 255), (10, 10, width, height), 2)


def draw_ui(surface, font, is_running, ball_detected, fps):
    """Рисует UI-элементы (текст, инструкции).

    Параметры:
        surface: pygame.Surface
        font: pygame.font.Font
        is_running: bool — запущена ли симуляция
        ball_detected: bool — найден ли шарик на камере
        fps: float — текущий FPS
    """
    # Инструкции
    lines = [
        "[ПРОБЕЛ] — запуск / сброс",
        "[Q] — выход",
        f"FPS: {fps:.1f}",
        f"Симуляция: {'▶ ЗАПУЩЕНА' if is_running else '⏸ ОСТАНОВЛЕНА'}",
        f"Шарик на камере: {'✓ НАЙДЕН' if ball_detected else '✗ НЕ НАЙДЕН'}",
    ]

    y_offset = 10
    for line in lines:
        text = font.render(line, True, (200, 200, 220))
        surface.blit(text, (surface.get_width() - text.get_width() - 10, y_offset))
        y_offset += 22

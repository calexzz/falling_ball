import sys
import pygame
import cv2

from camera import open_camera, release_camera, read_frame, detect_surfaces, find_ball
from physics import PhysicsEngine
from projector import draw_camera_background, draw_ball, draw_platforms, draw_ui


# ── Настройки ───────────────────────────────────────────────────
CAMERA_INDEX = 0          # индекс камеры (0 — первая доступная)
WIDTH, HEIGHT = 800, 600  # разрешение окна (должно совпадать с камерой)
FPS = 60                  # частота кадров


def surfaces_to_contours(surfaces):
    """Преобразует формат camera.py в формат physics.py.

    camera.py  возвращает: [(x1, y1, x2, y2), ...]
    physics.py принимает:  [((x1,y1), (x2,y2)), ...]
    """
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in surfaces]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Falling Ball — Камера + Шарик")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Инициализация камеры
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as e:
        print(f"Ошибка камеры: {e}")
        print("Запуск в режиме без камеры...")
        cap = None

    # Инициализация физики
    engine = PhysicsEngine(WIDTH, HEIGHT)

    # Состояние
    contours = []       # текущие платформы
    frame = None        # текущий кадр с камеры
    ball_detected = False
    running = True

    print("=" * 50)
    print("Falling Ball — запущен")
    print("[ПРОБЕЛ] — запуск / сброс симуляции")
    print("[Q]      — выход")
    print("=" * 50)

    while running:
        # ── События ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_SPACE:
                    # Пробел: запуск если остановлено, сброс если запущено
                    if engine.is_running():
                        engine.reset()
                        print("[СБРОС] Шарик возвращён на старт")
                    else:
                        engine.start()
                        print("[СТАРТ] Симуляция запущена!")

        # ── Получаем данные с камеры ─────────────────────────────
        if cap is not None:
            frame = read_frame(cap)
            if frame is not None:
                # Детекция платформ (линий)
                surfaces = detect_surfaces(frame)
                contours = surfaces_to_contours(surfaces)
                engine.set_platforms(contours)

                # Детекция шарика (для отображения статуса)
                ball_pos = find_ball(frame)
                ball_detected = ball_pos is not None

        # ── Шаг физики ───────────────────────────────────────────
        engine.step()
        state = engine.get_state()

        # ── Отрисовка ────────────────────────────────────────────
        # 1. Фон — изображение с камеры
        draw_camera_background(screen, frame)

        # 2. Платформы (поверх камеры)
        draw_platforms(screen, contours)

        # 3. Шарик (поверх всего)
        draw_ball(screen, state)

        # 4. UI (самый верхний слой)
        current_fps = clock.get_fps()
        draw_ui(screen, font, engine.is_running(), ball_detected, current_fps)

        # Обновляем экран
        pygame.display.flip()
        clock.tick(FPS)

    # ── Завершение ─────────────────────────────────────────────
    if cap is not None:
        release_camera(cap)
    pygame.quit()
    cv2.destroyAllWindows()
    print("Выход.")
    sys.exit()


if __name__ == "__main__":
    main()

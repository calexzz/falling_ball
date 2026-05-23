import pygame
import math
import sys

from camera import open_camera, release_camera, read_frame, detect_surfaces
from physics import PhysicsEngine


# ── Настройки ───────────────────────────────────────────────────
CAMERA_INDEX = 1
WIDTH, HEIGHT = 800, 600
FPS = 60

# Цвета (временная отрисовка — до подключения модуля проектора)
BG_COLOR       = (0,  0,  0)
BALL_COLOR     = (220, 90,  50)
PLATFORM_COLOR = (80, 140,  80)
TEXT_COLOR     = (200, 200, 220)


def surfaces_to_contours(surfaces):
    """
    camera.py  возвращает: [(x1, y1, x2, y2), ...]
    physics.py принимает:  [((x1,y1), (x2,y2)), ...]
    """
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in surfaces]


def draw_ball(surface, state):
    """Временная отрисовка шарика — пока нет модуля проектора."""
    cx, cy = int(state.x), int(state.y)
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), state.radius)
    pygame.draw.circle(surface, (255, 160, 100), (cx, cy), state.radius, 2)
    # маркер вращения
    mx = cx + int((state.radius - 4) * math.cos(state.angle))
    my = cy + int((state.radius - 4) * math.sin(state.angle))
    pygame.draw.circle(surface, (50, 20, 10), (mx, my), 4)


def draw_platforms(surface, contours):
    """Временная отрисовка платформ — пока нет модуля проектора."""
    for (x1, y1), (x2, y2) in contours:
        pygame.draw.line(surface, PLATFORM_COLOR, (x1, y1), (x2, y2), 8)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x1, y1), 4)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x2, y2), 4)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Физика шарика")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Arial", 18)

    # Инициализация камеры и физики
    cap    = open_camera(CAMERA_INDEX)
    engine = PhysicsEngine(WIDTH, HEIGHT)

    contours = []   # текущие платформы (обновляются с камеры каждый кадр)

    while True:
        # ── События
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                release_camera(cap)
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    release_camera(cap)
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    if engine.is_running():
                        engine.reset()
                    else:
                        engine.start()

        # ── Получаем данные с камеры
        frame = read_frame(cap)
        if frame is not None:
            surfaces = detect_surfaces(frame)          # [(x1,y1,x2,y2), ...]
            contours = surfaces_to_contours(surfaces)  # [((x1,y1),(x2,y2)), ...]
            engine.set_platforms(contours)             # передаём в физику

        # ── Шаг физики ───────────────────────────────────────────
        engine.step()
        state = engine.get_state()   # BallState → для проектора

        # ── Отрисовка (временная — заменит модуль проектора) ─────
        screen.fill(BG_COLOR)
        draw_platforms(screen, contours)
        draw_ball(screen, state)

        flipped = pygame.transform.flip(screen, True, False)
        screen.blit(flipped, (0, 0))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()

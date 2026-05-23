import pygame
import math
import sys

from camera import open_camera, release_camera, read_frame, detect_surfaces, find_ball
from physics import PhysicsEngine


# ── Настройки ───────────────────────────────────────────────────
CAMERA_INDEX = 0
WIDTH, HEIGHT = 800, 600
FPS = 60

BG_COLOR       = (0,   0,   0)
BALL_COLOR     = (220, 90,  50)
PLATFORM_COLOR = (80, 140,  80)


def surfaces_to_contours(surfaces):
    """
    Конвертирует формат линий camera.py → physics.py.

    Параметры:
        surfaces (list[tuple[int,int,int,int]]): [(x1,y1,x2,y2), ...]

    Возвращает:
        list[tuple[tuple,tuple]]: [((x1,y1),(x2,y2)), ...]
    """
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in surfaces]


def cam_to_screen(pos, cam_w, cam_h, scr_w=WIDTH, scr_h=HEIGHT):
    """
    Масштабирует координаты из пространства камеры в пространство экрана.

    Параметры:
        pos (tuple[int, int]): координаты (x, y) в пикселях камеры.
        cam_w (int): ширина кадра камеры в пикселях.
        cam_h (int): высота кадра камеры в пикселях.
        scr_w (int): ширина экрана/проектора.
        scr_h (int): высота экрана/проектора.

    Возвращает:
        tuple[int, int]: координаты (x, y) в пикселях экрана.
    """
    x = int(pos[0] * scr_w / cam_w)
    y = int(pos[1] * scr_h / cam_h)
    return x, y


def draw_ball(surface, state):
    """
    Отрисовывает шарик с маркером вращения.

    Параметры:
        surface (pygame.Surface): поверхность для рисования.
        state (BallState): текущее состояние шарика из physics.py.
    """
    cx, cy = int(state.x), int(state.y)
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), state.radius)
    pygame.draw.circle(surface, (255, 160, 100), (cx, cy), state.radius, 2)
    mx = cx + int((state.radius - 4) * math.cos(state.angle))
    my = cy + int((state.radius - 4) * math.sin(state.angle))
    pygame.draw.circle(surface, (50, 20, 10), (mx, my), 4)


def draw_platforms(surface, contours):
    """
    Отрисовывает платформы (линии с доски).

    Параметры:
        surface (pygame.Surface): поверхность для рисования.
        contours (list[tuple[tuple,tuple]]): [((x1,y1),(x2,y2)), ...]
    """
    for (x1, y1), (x2, y2) in contours:
        pygame.draw.line(surface, PLATFORM_COLOR, (x1, y1), (x2, y2), 8)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x1, y1), 4)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x2, y2), 4)


def draw_camera_ball(surface, pos):
    """
    Отрисовывает позицию шарика найденного камерой (для отладки).

    Параметры:
        surface (pygame.Surface): поверхность для рисования.
        pos (tuple[int, int] | None): координаты центра или None.
    """
    if pos is None:
        return
    pygame.draw.circle(surface, (0, 200, 255), pos, 8, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Физика шарика")
    clock = pygame.time.Clock()

    cap    = open_camera(CAMERA_INDEX)
    engine = PhysicsEngine(WIDTH, HEIGHT)

    contours   = []
    cam_ball   = None  # позиция шарика по камере (для отладки)
    cam_w, cam_h = WIDTH, HEIGHT  # обновляется после первого кадра

    while True:
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

        frame = read_frame(cap)
        if frame is not None:
            cam_h, cam_w = frame.shape[:2]

            surfaces = detect_surfaces(frame)
            contours = surfaces_to_contours(surfaces)
            engine.set_platforms(contours)

            # raw = find_ball(frame)
            # cam_ball = cam_to_screen(raw, cam_w, cam_h) if raw else None

        engine.step()
        state = engine.get_state()

        screen.fill(BG_COLOR)
        draw_platforms(screen, contours)
        draw_ball(screen, state)
        draw_camera_ball(screen, cam_ball)

        flipped = pygame.transform.flip(screen, True, False)
        screen.blit(flipped, (0,0))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()

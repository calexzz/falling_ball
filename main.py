import pygame
import math
import sys
import cv2
import numpy as np

from camera import open_camera, release_camera, read_frame, detect_surfaces, find_ball
from physics import PhysicsEngine


CAMERA_INDEX = 1
WIDTH, HEIGHT = 1024, 768
FPS = 60

BG_COLOR       = (0,   0,   0)
BALL_COLOR     = (220, 90,  50)
PLATFORM_COLOR = (80, 140,  80)

CALIB_PROJ = np.float32([
    [100,        100       ],
    [WIDTH-100,  100       ],
    [WIDTH-100,  HEIGHT-100],
    [100,        HEIGHT-100],
])

LABELS = ["1-верх-лево", "2-верх-право", "3-низ-право", "4-низ-лево"]

H_CAM_TO_PROJ = None


# ── Гомография ────────────────────────────────────────────────────────────────

def build_homography(cam_pts):
    src = np.float32(cam_pts)
    H, _ = cv2.findHomography(src, CALIB_PROJ)
    return H


def cam_contours_to_proj(contours, H):
    if not contours or H is None:
        return []
    pts = []
    for (x1, y1), (x2, y2) in contours:
        pts.append([x1, y1])
        pts.append([x2, y2])
    src = np.float32(pts).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    result = []
    for i in range(0, len(dst), 2):
        p1 = (int(dst[i][0]),   int(dst[i][1]))
        p2 = (int(dst[i+1][0]), int(dst[i+1][1]))
        result.append((p1, p2))
    return result


def surfaces_to_contours(surfaces):
    return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in surfaces]


# ── Отрисовка проектора ───────────────────────────────────────────────────────

def draw_ball(surface, state):
    cx, cy = int(state.x), int(state.y)
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), state.radius)
    pygame.draw.circle(surface, (255, 160, 100), (cx, cy), state.radius, 2)
    mx = cx + int((state.radius - 4) * math.cos(state.angle))
    my = cy + int((state.radius - 4) * math.sin(state.angle))
    pygame.draw.circle(surface, (50, 20, 10), (mx, my), 4)


def draw_platforms(surface, contours):
    for (x1, y1), (x2, y2) in contours:
        pygame.draw.line(surface, PLATFORM_COLOR, (x1, y1), (x2, y2), 8)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x1, y1), 4)
        pygame.draw.circle(surface, PLATFORM_COLOR, (x2, y2), 4)


def draw_calibration_screen(surface):
    """Чёрный фон + 4 ярко-жёлтые точки."""
    surface.fill((0, 0, 0))
    for px, py in CALIB_PROJ:
        pygame.draw.circle(surface, (255, 255, 255), (int(px), int(py)), 22)


# ── Отладочное окно ───────────────────────────────────────────────────────────

def build_debug_frame(cam_frame, raw_contours, calibration_mode,
                      clicked_pts, calibrated):
    out = cam_frame.copy()
    h, w = out.shape[:2]

    # Контуры с камеры
    for (x1, y1), (x2, y2) in raw_contours:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Уже отмеченные точки
    for i, (cx, cy) in enumerate(clicked_pts):
        cv2.circle(out, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(out, (cx, cy), 12, (255, 255, 255), 1)
        cv2.putText(out, LABELS[i], (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Подсказка — какую точку ставить следующей
    if calibration_mode and len(clicked_pts) < 4:
        next_label = LABELS[len(clicked_pts)]
        cv2.putText(out, f"Кликни: {next_label}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    elif calibrated:
        cv2.putText(out, "OK — гомография построена", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)

    # Панель снизу
    panel_h = 100
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)

    lines = [
        ("C      — режим калибровки (кликай 4 точки по углам доски)", (200, 200, 200)),
        ("R      — сбросить точки и начать заново",                   (200, 200, 200)),
        ("SPACE  — старт / сброс шарика",                             (200, 200, 200)),
        ("F11    — полный экран проектора  |  Q/ESC — выход",         (200, 200, 200)),
    ]
    y0 = h - panel_h + 20
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    # Счётчик точек
    cnt_color = (0, 255, 255) if calibration_mode else (60, 60, 60)
    cv2.putText(out, f"{len(clicked_pts)}/4", (w - 80, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cnt_color, 2)

    return out


# ── Обработчик кликов мыши в отладочном окне ─────────────────────────────────

_mouse_click = None

def on_mouse(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = (x, y)


# ── Главный цикл ──────────────────────────────────────────────────────────────

def main():
    global H_CAM_TO_PROJ, _mouse_click

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Проектор")
    clock = pygame.time.Clock()

    cap    = open_camera(CAMERA_INDEX)
    engine = PhysicsEngine(WIDTH, HEIGHT)

    raw_contours   = []
    proj_contours  = []
    last_cam_frame = None
    calibration_mode = False
    fullscreen       = False
    calibrated       = False
    clicked_pts      = []   # до 4 точек в координатах камеры

    debug_window = "Отладка"
    cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(debug_window, 720, 540)
    cv2.setMouseCallback(debug_window, on_mouse)

    while True:
        # ── pygame события ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                release_camera(cap)
                pygame.quit()
                cv2.destroyAllWindows()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    release_camera(cap)
                    pygame.quit()
                    cv2.destroyAllWindows()
                    sys.exit()

                if event.key == pygame.K_SPACE:
                    if engine.is_running():
                        engine.reset()
                        print("[BALL] сброс")
                    else:
                        engine.start()
                        print("[BALL] старт")

                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    flags = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
                    print(f"[WIN] {'полный экран' if fullscreen else 'окно'}")

                if event.key == pygame.K_c:
                    calibration_mode = not calibration_mode
                    print(f"[MODE] {'калибровка' if calibration_mode else 'симуляция'}")

                if event.key == pygame.K_r:
                    clicked_pts = []
                    calibrated  = False
                    H_CAM_TO_PROJ = None
                    print("[CALIB] точки сброшены")

            if event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        # ── Клик мыши в отладочном окне ──────────────────────────────────────
        if _mouse_click is not None and calibration_mode:
            if len(clicked_pts) < 4:
                cx, cy = _mouse_click
                # Отладочное окно может быть другого размера — пересчитываем
                # в координаты кадра камеры
                if last_cam_frame is not None:
                    dh, dw = last_cam_frame.shape[:2]
                    win_rect = cv2.getWindowImageRect(debug_window)
                    win_w, win_h = win_rect[2], win_rect[3]
                    if win_w > 0 and win_h > 0:
                        cx = int(cx * dw / win_w)
                        cy = int(cy * dh / win_h)
                clicked_pts.append((cx, cy))
                print(f"[CALIB] точка {len(clicked_pts)}: ({cx}, {cy})  ({LABELS[len(clicked_pts)-1]})")

                if len(clicked_pts) == 4:
                    H_CAM_TO_PROJ = build_homography(clicked_pts)
                    calibrated = True
                    print("[CALIB] OK — гомография построена")
            _mouse_click = None

        # ── Камера ───────────────────────────────────────────────────────────
        frame = read_frame(cap)
        if frame is not None:
            last_cam_frame = frame
            if not calibration_mode:
                surfaces = detect_surfaces(frame)
                raw_contours  = surfaces_to_contours(surfaces)
                prev_len = len(proj_contours)
                proj_contours = cam_contours_to_proj(raw_contours, H_CAM_TO_PROJ)
                if len(proj_contours) != prev_len:
                    print(f"[SURF] контуров: {len(proj_contours)}")
                engine.set_platforms(proj_contours)

        # ── Физика ───────────────────────────────────────────────────────────
        engine.step()
        state = engine.get_state()

        # ── Проектор ─────────────────────────────────────────────────────────
        if calibration_mode:
            draw_calibration_screen(screen)
        else:
            screen.fill(BG_COLOR)
            draw_platforms(screen, proj_contours)
            draw_ball(screen, state)

        pygame.display.flip()
        clock.tick(FPS)

        # ── Отладочное окно ───────────────────────────────────────────────────
        if last_cam_frame is not None:
            dbg = build_debug_frame(last_cam_frame, raw_contours,
                                    calibration_mode, clicked_pts, calibrated)
            cv2.imshow(debug_window, dbg)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

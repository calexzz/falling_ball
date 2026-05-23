from dataclasses import dataclass
import pymunk


@dataclass
class BallState:
    """Состояние шарика — передаётся в модуль отрисовки."""
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    radius: int


BALL_RADIUS = 18
BALL_MASS = 1.0
BALL_ELASTICITY = 0.45
BALL_FRICTION  = 0.6

PLATFORM_ELASTICITY = 0.35
PLATFORM_FRICTION   = 0.7

GRAVITY = 900
FPS = 60


class PhysicsEngine:
    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._running = False

        self._space = pymunk.Space()
        self._space.gravity = (0, GRAVITY)
        self._space.damping = 0.995

        self._platform_shapes: list[pymunk.Segment] = []
        self._ball_body, self._ball_shape = self._create_ball()

    def set_platforms(self, contours: list[tuple]) -> None:
        """Принимает контуры с камеры и строит физические платформы."""
        for shape in self._platform_shapes:
            self._space.remove(shape)
        self._platform_shapes.clear()

        # Строим новые
        for point_a, point_b in contours:
            seg = pymunk.Segment(
                self._space.static_body,
                point_a, point_b,
                radius=6
            )
            seg.elasticity = PLATFORM_ELASTICITY
            seg.friction   = PLATFORM_FRICTION
            self._space.add(seg)
            self._platform_shapes.append(seg)

    def start(self) -> None:
        """Запускает симуляцию (шарик начинает падать)."""
        self._running = True

    def stop(self) -> None:
        """Останавливает симуляцию (шарик замирает)."""
        self._running = False

    def reset(self) -> None:
        """Возвращает шарик на стартовую позицию."""
        body = self._ball_body
        body.position         = (self.width // 2, 60)
        body.velocity         = (0, 0)
        body.angular_velocity = 0
        body.angle            = 0
        self._running = False

    def step(self) -> None:
        """
        Один шаг симуляции (вызывать каждый кадр).
        Автоматически сбрасывает шарик при выходе за экран.
        """
        if not self._running:
            return

        self._space.step(1 / FPS)

        x, y = self._ball_body.position
        out_of_bounds = (
            y > self.height + 80 or
            x < -80 or
            x > self.width + 80
        )
        if out_of_bounds:
            self.reset()

    def get_state(self) -> BallState:
        """
        Возвращает текущее состояние шарика.
        Вызывать каждый кадр после step().
        """
        body = self._ball_body
        return BallState(
            x      = body.position.x,
            y      = body.position.y,
            vx     = body.velocity.x,
            vy     = body.velocity.y,
            angle  = body.angle,
            radius = BALL_RADIUS,
        )

    def is_running(self) -> bool:
        """True если симуляция запущена."""
        return self._running

    def _create_ball(self) -> tuple[pymunk.Body, pymunk.Circle]:
        moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        body   = pymunk.Body(BALL_MASS, moment)
        body.position = (self.width // 2, 60)

        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = BALL_ELASTICITY
        shape.friction   = BALL_FRICTION

        self._space.add(body, shape)
        return body, shape

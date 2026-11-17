from world.anatomy.base import Mouth
from world.datatypes import Motion, Posture, Sound
import numpy as np

JAW, LIFT, PLACE, BREATH, VOICE = range(5)


class VoiceBox(Mouth):

    REST = np.array([0.25, 0.0, 0.5, 0.0, 0.0])
    MASS = np.array([1.0, 0.3, 1.5, 0.5, 0.2])
    PERIOD = np.array([10.0, 5.0, 14.0, 8.0, 4.0])
    DAMPING = np.array([0.3, 0.7, 0.8, 0.8, 0.9])

    SPRING_K = MASS * (2 * np.pi / PERIOD) ** 2
    FRICTION = 2 * DAMPING * np.sqrt(SPRING_K * MASS)
    MAX_FORCE = 1.5 * SPRING_K
    STRENGTH = MAX_FORCE / MAX_FORCE.sum()
    WALL = 3.0
    SUBSTEPS = 4

    # The controller that drives the mouth to a target.
    DRIVEN_PERIOD = np.minimum(PERIOD, 7.0)                   # slow parts are driven as if their swing took 7 frames, quick parts are left alone
    STIFFNESS = MASS * (2 * np.pi / DRIVEN_PERIOD) ** 2       # same formula as SPRING_K, with the driven period
    GAIN = STIFFNESS / SPRING_K - 1                           # how much harder than the bare spring it pulls towards the target
    BRAKE = 2 * np.sqrt(STIFFNESS * MASS) - FRICTION          # extra friction that stops it overshooting: critical damping = 2 * sqrt(STIFFNESS * MASS)

    BANDS = 8
    CENTRES = (np.arange(BANDS) + 0.5) / BANDS

    @property
    def rest(self) -> Posture:
        return Posture(self.REST.copy(), np.zeros(5))

    def steer(self, state: Posture, target: Posture) -> Motion:
        hold = self.SPRING_K * (target.position - self.REST)                    # what it takes to stay at the target against the spring
        pull = self.GAIN * self.SPRING_K * (target.position - state.position)  # harder the further away it still is
        brake = self.BRAKE * state.velocity                                     # slow down on the way in
        return Motion.make(np.clip((hold + pull - brake) / self.MAX_FORCE, -1.0, 1.0))

    def arrived(self, state: Posture, target: Posture, tolerance: float) -> bool:
        return bool(np.all(np.abs(state.position - target.position) < tolerance) and np.all(np.abs(state.velocity) < tolerance))

    def move(self, state: Posture, motion: Motion) -> tuple[Posture, float]:
        share = np.clip(motion.toArray(), -1.0, 1.0)
        push = share * self.MAX_FORCE
        x, v = state.position, state.velocity
        dt = 1.0 / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            wall = self.WALL * self.SPRING_K * (np.maximum(-x, 0.0) - np.maximum(x - 1.0, 0.0))
            a = (push - self.SPRING_K * (x - self.REST) - self.FRICTION * v + wall) / self.MASS
            v = v + dt * a
            x = x + dt * v

        gap, breath = self.opening(x), float(np.clip(x[BREATH], 0.0, 1.0))
        shut = np.exp(-(gap / 0.03) ** 2)
        burst = state.pressure * np.tanh(8.0 * gap)
        pressure = float(np.clip(state.pressure + 0.5 * breath * shut - 0.8 * burst - 0.05 * state.pressure, 0.0, 1.0))

        effort = float((share ** 2 * self.STRENGTH).sum())
        return Posture(x, v, pressure, float(burst)), effort

    def sound(self, state: Posture) -> Sound:
        x = state.position
        gap = self.opening(x)
        place, breath, voice = np.clip(x[[PLACE, BREATH, VOICE]], 0.0, 1.0)

        gets_out = gap / (gap + 0.15)
        voiced = breath * voice * gets_out
        whisper = 0.1 * breath * (1 - voice) * gets_out
        hiss = breath * np.exp(-((gap - 0.08) / 0.06) ** 2) * np.tanh(gap / 0.02)

        f1 = 0.1 + 0.45 * np.tanh(2.0 * gap)
        f2 = 0.9 - 0.5 * place
        peak = lambda centre, width: np.exp(-((self.CENTRES - centre) / width) ** 2)
        tone = voiced * (peak(f1, 0.09) + 0.8 * peak(f2, 0.09)) + whisper * (peak(f1, 0.2) + 0.8 * peak(f2, 0.2))
        noise = (hiss + state.burst) * peak(0.95 - 0.6 * place, 0.15)
        return Sound.make(tone + noise)

    @staticmethod
    def opening(position: np.ndarray) -> float:
        return float(np.logaddexp(0.0, 30.0 * (position[JAW] - position[LIFT])) / 30.0)

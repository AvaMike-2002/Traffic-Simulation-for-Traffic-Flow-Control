import os
import pickle
import random
import sys
import time
import csv
from datetime import datetime
from collections import defaultdict
from math import cos, sin

import numpy as np
import pygame

# =========================
# Constants
# =========================
WIDTH = 1200
HEIGHT = 900
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
FPS = 60

# Different lane counts for different directions
LANE_COUNT_NS = 5  # North-South roads have 5 lanes (N0-N4, S0-S4)
LANE_COUNT_EW = 4  # East-West roads have 4 lanes (E0-E3, W0-W3)

LANE_WIDTH = 25

# Calculate road half widths based on lane counts
ROAD_HALF_WIDTH_NS = (LANE_COUNT_NS * LANE_WIDTH) // 2
ROAD_HALF_WIDTH_EW = (LANE_COUNT_EW * LANE_WIDTH) // 2

INTERSECTION_BOX = max(ROAD_HALF_WIDTH_NS, ROAD_HALF_WIDTH_EW) + 30

CAR_LENGTH = 20
CAR_WIDTH = 14

CAR_BASE_SPEED = 3.5
CAR_TURN_SPEED = 2.5
CAR_ACCEL = 0.15
CAR_BRAKE = 0.25
SPAWN_COOLDOWN = 6

# Traffic light timing
GREEN_DURATION = 5.0  # seconds
YELLOW_DURATION = 2.0  # seconds

# Car colors
CAR_COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
]
AGENT_COLOR = (50, 200, 50)

TURN_RADIUS = 70
CURVE_POINTS = 40

# Lane turn mapping - Spawning lanes only (N0,N1 only - no N2, S0,S1 only - no S2)
LANE_TURN_MAPPING = {
    "N": {0: ["straight"], 1: ["straight"], 2: [], 3: [], 4: []},
    "S": {0: ["straight"], 1: ["left"], 2: [], 3: [], 4: []},
    "E": {2: ["straight"], 3: ["straight"]},
    "W": {2: ["straight"], 3: ["straight"]}
}

# Map turn types to target lanes
TURN_TARGET_LANE = {
    "N": {"straight": 4},
    "S": {"straight": 0, "left": 4},
    "E": {"straight": 0},
    "W": {"straight": 0}
}

# Spawning lanes (incoming traffic) - N2 and S2 removed
SPAWNING_LANES = {
    "N": [0, 1],  # N0, N1 only (N2 removed)
    "S": [0, 1],  # S0, S1 only (S2 removed)
    "E": [2, 3],  # E2, E3
    "W": [2, 3]  # W2, W3
}

# Receiving lanes (outgoing traffic)
RECEIVING_LANES = {
    "N": [3, 4],
    "S": [3, 4],
    "E": [0, 1],
    "W": [0, 1]
}

DIRECTION_MAPPING = {
    "N": {"left": "W", "straight": "S", "right": "E"},
    "S": {"left": "E", "straight": "N", "right": "W"},
    "E": {"left": "N", "straight": "W", "right": "S"},
    "W": {"left": "S", "straight": "E", "right": "N"}
}

LANE_DEPARTURE_THRESHOLD = LANE_WIDTH * 0.8

CSV_FILENAME = f"traffic_rl_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def get_lane_count(direction: str) -> int:
    return LANE_COUNT_NS if direction in ["N", "S"] else LANE_COUNT_EW


def get_road_half_width(direction: str) -> int:
    return ROAD_HALF_WIDTH_NS if direction in ["N", "S"] else ROAD_HALF_WIDTH_EW


# =========================
# CSV Logger Class
# =========================
class CSVLogger:
    def __init__(self, filename=CSV_FILENAME):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'step', 'epoch', 'decision',
            'action', 'phase', 'reward', 'cumulative_reward',
            'cars_n', 'cars_s', 'cars_w', 'cars_e',
            'completed_cars', 'collisions', 'red_violations',
            'lane_departures', 'total_cars', 'epsilon'
        ])
        self.file.flush()
        print(f"[✓] CSV logging to {filename}")

    def log(self, step, epoch, decision, action, phase, reward, cumulative_reward,
            cars_waiting, completed_cars, collisions, red_violations,
            lane_departures, total_cars, epsilon):
        self.writer.writerow([
            datetime.now().isoformat(), step, epoch, decision,
            action, phase, f"{reward:.4f}", f"{cumulative_reward:.4f}",
            cars_waiting.get('N', 0), cars_waiting.get('S', 0),
            cars_waiting.get('W', 0), cars_waiting.get('E', 0),
            completed_cars, collisions, red_violations,
            lane_departures, total_cars, f"{epsilon:.6f}"
        ])
        self.file.flush()

    def log_epoch_summary(self, epoch, avg_reward, best_reward, epsilon, q_table_size, loss):
        with open(f"epoch_summary_{datetime.now().strftime('%Y%m%d')}.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['epoch', 'avg_reward', 'best_reward', 'epsilon', 'q_table_size', 'loss', 'timestamp'])
            writer.writerow([epoch, f"{avg_reward:.4f}", f"{best_reward:.4f}",
                             f"{epsilon:.6f}", q_table_size, f"{loss:.6f}",
                             datetime.now().isoformat()])

    def close(self):
        self.file.close()
        print(f"[✓] CSV file saved: {self.filename}")


# =========================
# Utility Functions
# =========================
def direction_vector(direction: str):
    if direction == "N": return np.array([0.0, 1.0])
    if direction == "S": return np.array([0.0, -1.0])
    if direction == "W": return np.array([1.0, 0.0])
    return np.array([-1.0, 0.0])


def get_turn_path(start_dir, turn_type):
    cx, cy = CENTER_X, CENTER_Y
    points = []
    radius = TURN_RADIUS

    if turn_type == "left":
        if start_dir == "N":
            for t in np.linspace(np.pi / 2, np.pi, CURVE_POINTS):
                points.append((cx - radius * np.cos(t), cy - radius * np.sin(t)))
        elif start_dir == "S":
            for t in np.linspace(np.pi, 3 * np.pi / 2, CURVE_POINTS):
                points.append((cx + radius * np.cos(t), cy + radius * np.sin(t)))
        elif start_dir == "E":
            for t in np.linspace(-np.pi / 2, 0, CURVE_POINTS):
                points.append((cx + radius * np.cos(t), cy - radius * np.sin(t)))
        elif start_dir == "W":
            for t in np.linspace(np.pi / 2, np.pi, CURVE_POINTS):
                points.append((cx - radius * np.cos(t), cy + radius * np.sin(t)))
    elif turn_type == "right":
        if start_dir == "N":
            for t in np.linspace(-np.pi / 2, 0, CURVE_POINTS):
                points.append((cx + radius * np.cos(t), cy - radius * np.sin(t)))
        elif start_dir == "S":
            for t in np.linspace(np.pi / 2, np.pi, CURVE_POINTS):
                points.append((cx - radius * np.cos(t), cy + radius * np.sin(t)))
        elif start_dir == "E":
            for t in np.linspace(np.pi, 3 * np.pi / 2, CURVE_POINTS):
                points.append((cx + radius * np.cos(t), cy + radius * np.sin(t)))
        elif start_dir == "W":
            for t in np.linspace(-np.pi / 2, 0, CURVE_POINTS):
                points.append((cx - radius * np.cos(t), cy - radius * np.sin(t)))
    return points


def lane_center_for(direction: str, lane: int):
    lane_count = get_lane_count(direction)
    lane_offset = (lane - (lane_count - 1) / 2.0) * LANE_WIDTH

    if direction == "N":
        return CENTER_X + lane_offset, None
    if direction == "S":
        return CENTER_X - lane_offset, None
    if direction == "W":
        return None, CENTER_Y + lane_offset
    return None, CENTER_Y - lane_offset


def stop_line_for(direction: str, lane: int):
    lane_x, lane_y = lane_center_for(direction, lane)
    road_half_width = get_road_half_width(direction)

    if direction == "N":
        return lane_x, CENTER_Y - road_half_width - 5
    if direction == "S":
        return lane_x, CENTER_Y + road_half_width + 5
    if direction == "W":
        return CENTER_X - road_half_width - 5, lane_y
    return CENTER_X + road_half_width + 5, lane_y


def get_lane_deviation(car):
    lane_x, lane_y = lane_center_for(car.move_direction, car.lane)
    if lane_x is not None:
        return abs(car.x - lane_x)
    return abs(car.y - lane_y)


def distance_to_intersection(car):
    road_half_width = get_road_half_width(car.direction)
    if car.direction == "N":
        return abs(car.y - (CENTER_Y - road_half_width))
    elif car.direction == "S":
        return abs(car.y - (CENTER_Y + road_half_width))
    elif car.direction == "W":
        return abs(car.x - (CENTER_X - road_half_width))
    else:
        return abs(car.x - (CENTER_X + road_half_width))


# =========================
# Core Classes
# =========================
class TrafficLight:
    def __init__(self, direction: str, lane: int, turn_type: str):
        self.direction = direction
        self.lane = lane
        self.turn_type = turn_type
        self.state = "GREEN"
        self.position = stop_line_for(direction, lane)
        self.timer = 0.0


class Car:
    def __init__(self, direction: str, lane: int, turn_intent: str, agent_car=False):
        self.direction = direction
        self.lane = lane
        self.turn_intent = turn_intent
        self.agent_car = agent_car
        self.destroyed = False
        self.violation = None

        self.target_direction = DIRECTION_MAPPING[direction][turn_intent]
        self.move_direction = direction
        self.heading = direction_vector(direction)

        self.speed = CAR_BASE_SPEED * 0.3
        self.max_speed = CAR_BASE_SPEED
        self.turn_speed = CAR_TURN_SPEED
        self.acc = CAR_ACCEL
        self.brake = CAR_BRAKE

        self.color = AGENT_COLOR if agent_car else random.choice(CAR_COLORS)
        self.has_turned = False
        self.has_crossed_intersection = False
        self.has_run_red = False
        self.id = random.randint(1000, 9999)

        self.turn_path = None
        self.path_index = 0

        if turn_intent != "straight":
            self.target_lane = TURN_TARGET_LANE[direction][turn_intent]
        else:
            if direction in ["N", "S"]:
                self.target_lane = 4
            else:
                self.target_lane = 0

        self.x, self.y = self._spawn_pos(direction, lane)
        self.collision_box = pygame.Rect(0, 0, CAR_LENGTH, CAR_WIDTH)
        print(f"[SPAWN] Car {self.id} at {direction}, lane {lane}, turn {turn_intent}")

    @staticmethod
    def _lane_offset(lane: int, direction: str):
        lane_count = get_lane_count(direction)
        return (lane - (lane_count - 1) / 2.0) * LANE_WIDTH

    def _spawn_pos(self, direction: str, lane: int):
        offset = self._lane_offset(lane, direction)

        if direction == "N":
            return CENTER_X + offset, -CAR_LENGTH - 10
        if direction == "S":
            return CENTER_X - offset, HEIGHT + CAR_LENGTH + 10
        if direction == "W":
            return -CAR_LENGTH - 10, CENTER_Y + offset
        return WIDTH + CAR_LENGTH + 10, CENTER_Y - offset

    def is_out_of_bounds(self):
        margin = 200
        return (self.x < -margin or self.x > WIDTH + margin or
                self.y < -margin or self.y > HEIGHT + margin)

    def is_before_stop_line(self):
        stop_x, stop_y = stop_line_for(self.direction, self.lane)
        if self.direction == "N": return self.y < stop_y - 5
        if self.direction == "S": return self.y > stop_y + 5
        if self.direction == "W": return self.x < stop_x - 5
        return self.x > stop_x + 5

    def is_near_stop_line(self):
        stop_x, stop_y = stop_line_for(self.direction, self.lane)
        threshold = 50
        if self.direction == "N": return abs(self.y - stop_y) < threshold
        if self.direction == "S": return abs(self.y - stop_y) < threshold
        if self.direction == "W": return abs(self.x - stop_x) < threshold
        return abs(self.x - stop_x) < threshold

    def has_crossed_stop_line(self):
        stop_x, stop_y = stop_line_for(self.direction, self.lane)
        if self.direction == "N": return self.y > stop_y
        if self.direction == "S": return self.y < stop_y
        if self.direction == "W": return self.x > stop_x
        return self.x < stop_x

    def is_in_intersection(self):
        margin = 40
        return (abs(self.x - CENTER_X) < INTERSECTION_BOX + margin and
                abs(self.y - CENTER_Y) < INTERSECTION_BOX + margin)

    def _init_turn_path(self):
        if self.turn_intent != "straight" and not self.has_turned:
            self.turn_path = get_turn_path(self.direction, self.turn_intent)
            self.path_index = 0

    def _follow_curved_path(self, dt):
        if self.turn_path and self.path_index < len(self.turn_path):
            target_x, target_y = self.turn_path[self.path_index]
            dx, dy = target_x - self.x, target_y - self.y
            distance = np.hypot(dx, dy)

            if distance < 3:
                self.path_index += 1
                if self.path_index >= len(self.turn_path):
                    self.has_turned = True
                    self.move_direction = self.target_direction
                    self.heading = direction_vector(self.target_direction)
                    self.lane = self.target_lane
                    self.turn_path = None
            else:
                steps = min(1.0, self.speed * dt * 3.0 / max(distance, 0.1))
                self.x += dx * steps
                self.y += dy * steps

    def _stay_in_lane(self):
        lane_x, lane_y = lane_center_for(self.move_direction, self.lane)
        strength = 0.15
        if lane_x is not None:
            self.x += (lane_x - self.x) * strength
        if lane_y is not None:
            self.y += (lane_y - self.y) * strength

    def _update_collision_box(self):
        horizontal = abs(self.heading[0]) > abs(self.heading[1])
        w, h = (CAR_LENGTH, CAR_WIDTH) if horizontal else (CAR_WIDTH, CAR_LENGTH)
        self.collision_box = pygame.Rect(int(self.x - w / 2), int(self.y - h / 2), w, h)

    def check_lane_departure(self):
        if not self.has_turned and self.turn_intent != "straight":
            return False
        return get_lane_deviation(self) > LANE_DEPARTURE_THRESHOLD

    def move(self, can_move: bool, dt: float, light_state=None):
        if self.destroyed:
            return

        turning = self.turn_intent != "straight" and not self.has_turned and self.is_in_intersection()
        target_speed = self.turn_speed if turning else self.max_speed

        if light_state == "RED" and not self.has_crossed_intersection and self.is_before_stop_line():
            self.speed = max(0, self.speed - self.brake * 2.0)
            can_move = False
        elif light_state == "YELLOW" and not self.has_crossed_intersection and self.is_before_stop_line():
            self.speed = max(0, self.speed - self.brake * 1.2)
            if self.is_near_stop_line() and self.speed < 2.0:
                can_move = False
        else:
            can_move = True

        if can_move and light_state != "RED":
            self.speed = min(target_speed, self.speed + self.acc)
        elif not can_move:
            self.speed = max(0.0, self.speed - self.brake)

        if turning and not self.has_turned:
            if not self.turn_path:
                self._init_turn_path()
            self._follow_curved_path(dt)
        else:
            self.x += self.heading[0] * self.speed
            self.y += self.heading[1] * self.speed

        self._stay_in_lane()
        self._update_collision_box()

        if self.check_lane_departure():
            self.violation = "LANE_DEPARTURE"
            self.destroyed = True
            return

        if not self.has_crossed_intersection and not self.is_in_intersection() and self.has_crossed_stop_line():
            self.has_crossed_intersection = True

    def draw(self, screen, highlight=False):
        if self.destroyed:
            w, h = CAR_LENGTH, CAR_WIDTH
            wreck_surface = pygame.Surface((w, h), pygame.SRCALPHA)
            wreck_surface.fill((80, 80, 80, 200))
            pygame.draw.rect(wreck_surface, (50, 50, 50), (0, 0, w, h), border_radius=3)
            pygame.draw.line(wreck_surface, (255, 0, 0), (0, 0), (w, h), 2)
            pygame.draw.line(wreck_surface, (255, 0, 0), (w, 0), (0, h), 2)
            screen.blit(wreck_surface, (int(self.x - w / 2), int(self.y - h / 2)))
            return

        color = AGENT_COLOR if highlight else self.color
        if self.move_direction in ["N", "S"]:
            w, h = CAR_WIDTH, CAR_LENGTH
        else:
            w, h = CAR_LENGTH, CAR_WIDTH

        car_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, color, (0, 0, w, h), border_radius=4)
        pygame.draw.rect(car_surface, (0, 0, 0), (0, 0, w, h), 2, border_radius=4)

        font = pygame.font.Font(None, 16)
        if self.turn_intent == "left" and not self.has_turned:
            arrow = "←"
        elif self.turn_intent == "right" and not self.has_turned:
            arrow = "→"
        else:
            arrow = "↓" if self.move_direction == "N" else "↑" if self.move_direction == "S" else "→" if self.move_direction == "E" else "←"
        text = font.render(arrow, True, (255, 255, 255))
        car_surface.blit(text, (w // 2 - 5, h // 2 - 8))

        if self.move_direction == "N":
            pygame.draw.circle(car_surface, (255, 255, 200), (w // 2, h - 4), 3)
        elif self.move_direction == "S":
            pygame.draw.circle(car_surface, (255, 255, 200), (w // 2, 4), 3)
        elif self.move_direction == "E":
            pygame.draw.circle(car_surface, (255, 255, 200), (w - 4, h // 2), 3)
        elif self.move_direction == "W":
            pygame.draw.circle(car_surface, (255, 255, 200), (4, h // 2), 3)

        screen.blit(car_surface, (int(self.x - w / 2), int(self.y - h / 2)))
        if self.agent_car:
            rl_text = font.render("RL", True, (255, 255, 0))
            screen.blit(rl_text, (int(self.x - 12), int(self.y - 22)))


class LaneQueue:
    def __init__(self, direction: str):
        self.direction = direction
        self.lane_count = get_lane_count(direction)
        self.queue = [[] for _ in range(self.lane_count)]

    def enqueue(self, car: Car):
        if car.lane < len(self.queue):
            self.queue[car.lane].append(car)
            self.queue[car.lane].sort(key=lambda c: distance_to_intersection(c))

    def dequeue(self, lane: int):
        if lane < len(self.queue) and self.queue[lane]:
            return self.queue[lane].pop(0)
        return None

    def get_front_car(self, car: Car):
        if car.lane >= len(self.queue):
            return None
        lane_list = self.queue[car.lane]
        if car not in lane_list:
            return None
        idx = lane_list.index(car)
        return lane_list[idx - 1] if idx > 0 else None


class CarsSpawner:
    def __init__(self, max_cars=50):
        self.cars = []
        self.max_cars = max_cars
        self.cooldown = 0
        self.agent_car = None

    def spawn_agent_car(self, lane_queues: dict):
        if self.agent_car and not self.agent_car.is_out_of_bounds() and not self.agent_car.destroyed:
            return
        direction = random.choice(["N", "S", "E", "W"])
        lane = random.choice(SPAWNING_LANES[direction])
        available_turns = LANE_TURN_MAPPING[direction].get(lane, [])
        if available_turns:
            turn_intent = random.choice(available_turns)
            self.agent_car = Car(direction=direction, lane=lane, turn_intent=turn_intent, agent_car=True)
            self.cars.append(self.agent_car)
            lane_queues[direction].enqueue(self.agent_car)

    def maybe_spawn_car(self, lane_queues: dict):
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if len(self.cars) >= self.max_cars:
            return
        direction = random.choice(["N", "S", "E", "W"])
        lane = random.choice(SPAWNING_LANES[direction])
        available_turns = LANE_TURN_MAPPING[direction].get(lane, [])
        if available_turns:
            turn_intent = random.choice(available_turns)
            car = Car(direction=direction, lane=lane, turn_intent=turn_intent)
            car.id = len(self.cars) + 1
            self.cars.append(car)
            lane_queues[direction].enqueue(car)
            self.cooldown = SPAWN_COOLDOWN

    def remove_destroyed_cars(self, lane_queues: dict):
        for car in self.cars[:]:
            if car.destroyed or car.is_out_of_bounds():
                if car.direction in lane_queues:
                    lane_queues[car.direction].dequeue(car.lane)
                self.cars.remove(car)
                if car == self.agent_car:
                    self.agent_car = None


class CarsController:
    def __init__(self, cars: list, traffic_lights: list):
        self.cars = cars
        self.traffic_lights = traffic_lights
        self.lane_queues = {d: LaneQueue(d) for d in ["N", "S", "W", "E"]}
        self.red_light_violations = 0
        self.lane_departures = 0

    def _check_collision(self, car1: Car, car2: Car):
        if car1.destroyed or car2.destroyed:
            return False
        return car1.collision_box.colliderect(car2.collision_box)

    def _too_close(self, car: Car, front_car: Car):
        if not front_car or front_car.destroyed:
            return False
        if car.direction in ["N", "S"]:
            distance = abs(car.y - front_car.y)
        else:
            distance = abs(car.x - front_car.x)

        relative_speed = max(0, car.speed - front_car.speed)
        base_distance = CAR_LENGTH * 1.5
        safe_distance = base_distance + car.speed * 2.0 + relative_speed * 1.5

        if car.is_near_stop_line():
            safe_distance += 15
        return distance < safe_distance

    def _get_relevant_traffic_light(self, car: Car):
        matching_lights = [tl for tl in self.traffic_lights
                           if tl.direction == car.direction and tl.lane == car.lane and tl.turn_type == car.turn_intent]
        return matching_lights[0] if matching_lights else None

    def update_cars_positions(self, screen, render_game=False):
        cars_waiting = {"N": 0, "S": 0, "W": 0, "E": 0}
        cars_completed = 0
        collisions = []

        for car in self.cars:
            if car.has_turned and car.turn_intent != "straight" and not car.destroyed:
                for lane_queue in self.lane_queues.values():
                    if car.lane < len(lane_queue.queue) and car in lane_queue.queue[car.lane]:
                        lane_queue.queue[car.lane].remove(car)
                car.direction = car.move_direction
                self.lane_queues[car.direction].enqueue(car)
                car.has_turned = False

        for i, car in enumerate(self.cars):
            if car.destroyed:
                continue

            should_stop = False

            front_car = None
            lane_list = self.lane_queues[car.direction].queue[car.lane]
            if car in lane_list:
                idx = lane_list.index(car)
                if idx > 0:
                    front_car = lane_list[idx - 1]

            if front_car and self._too_close(car, front_car):
                should_stop = True
                if car.speed > front_car.speed:
                    car.speed = max(front_car.speed, car.speed - car.brake * 1.5)

            relevant_tl = self._get_relevant_traffic_light(car)
            light_state = relevant_tl.state if relevant_tl else "GREEN"

            if relevant_tl and relevant_tl.state in ["RED", "YELLOW"] and car.is_before_stop_line():
                cars_waiting[car.direction] += 1
                if car.is_near_stop_line():
                    should_stop = True

            car.move(not should_stop, dt=1 / FPS, light_state=light_state)

            for other_car in self.cars[i + 1:]:
                if other_car.destroyed:
                    continue
                if self._check_collision(car, other_car):
                    collisions.append((car, other_car))
                    car.destroyed = True
                    other_car.destroyed = True
                    car.violation = "COLLISION"
                    other_car.violation = "COLLISION"

            if car.has_crossed_intersection and not car.destroyed:
                cars_completed += 1

            if render_game:
                car.draw(screen, highlight=car.agent_car)

        for car in self.cars:
            if not car.destroyed and not car.has_crossed_intersection and car.is_before_stop_line():
                cars_waiting[car.direction] += 1

        for car in self.cars:
            if car.violation == "RED_LIGHT":
                self.red_light_violations += 1
                car.violation = None
            elif car.violation == "LANE_DEPARTURE":
                self.lane_departures += 1
                car.violation = None

        return cars_waiting, cars_completed, collisions


# =========================
# Traffic Light Scheduler
# =========================
class TrafficLightScheduler:
    def __init__(self, traffic_lights: list[TrafficLight], cars: list[Car]):
        self.traffic_lights = traffic_lights
        self.cars = cars

        self.current_phase = 0
        self.phase_timer = 0.0
        self.state = "GREEN"
        self.last_update_time = time.time()

        self.n_lights = [tl for tl in traffic_lights if tl.direction == "N"]
        self.s_lights = [tl for tl in traffic_lights if tl.direction == "S"]
        self.e_lights = [tl for tl in traffic_lights if tl.direction == "E"]
        self.w_lights = [tl for tl in traffic_lights if tl.direction == "W"]

        self._set_ns_green()
        self._set_ew_red()
        print("[LIGHT] Initialized: NS GREEN, EW RED")

    def _set_ns_green(self):
        for tl in self.n_lights + self.s_lights:
            tl.state = "GREEN"
        print("[LIGHT] NS Green ON")

    def _set_ns_yellow(self):
        for tl in self.n_lights + self.s_lights:
            if tl.state == "GREEN":
                tl.state = "YELLOW"
        print("[LIGHT] NS Yellow ON")

    def _set_ns_red(self):
        for tl in self.n_lights + self.s_lights:
            tl.state = "RED"

    def _set_ew_green(self):
        for tl in self.e_lights + self.w_lights:
            tl.state = "GREEN"
        print("[LIGHT] EW Green ON")

    def _set_ew_yellow(self):
        for tl in self.e_lights + self.w_lights:
            if tl.state == "GREEN":
                tl.state = "YELLOW"
        print("[LIGHT] EW Yellow ON")

    def _set_ew_red(self):
        for tl in self.e_lights + self.w_lights:
            tl.state = "RED"

    def update(self, dt: float):
        self.phase_timer += dt

        if self.state == "GREEN":
            if self.phase_timer >= GREEN_DURATION:
                self.state = "YELLOW"
                self.phase_timer = 0.0

                if self.current_phase == 0:
                    self._set_ns_yellow()
                else:
                    self._set_ew_yellow()
                print(f"[LIGHT] Phase {self.current_phase} -> YELLOW")

        elif self.state == "YELLOW":
            if self.phase_timer >= YELLOW_DURATION:
                self.state = "GREEN"
                self.phase_timer = 0.0
                self.current_phase = 1 - self.current_phase

                if self.current_phase == 0:
                    self._set_ns_green()
                    self._set_ew_red()
                else:
                    self._set_ew_green()
                    self._set_ns_red()
                print(f"[LIGHT] -> Phase {self.current_phase} GREEN")

    def set_pending_phase(self, phase: int):
        if phase != self.current_phase and self.state == "GREEN":
            self.state = "YELLOW"
            self.phase_timer = 0.0
            if self.current_phase == 0:
                self._set_ns_yellow()
            else:
                self._set_ew_yellow()
            print(f"[LIGHT] RL forcing phase switch to {phase}")


# =========================
# Environment (RL)
# =========================
class TrafficEnv:
    def __init__(self, max_cars=50):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Traffic Intersection RL - Cleaned (No N2/S2)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.last_frame_time = time.time()

        self.max_cars = max_cars
        self.last_reward = 0.0
        self.last_action = None
        self.repeat_count = 0
        self.cars_waiting = {"N": 0, "S": 0, "W": 0, "E": 0}
        self.collision_count = 0
        self.step_count = 0
        self.completed_cars = 0
        self.epoch_reward = 0.0
        self.decision_reward = 0.0

        # Create traffic lights for spawning lanes only
        self.traffic_lights = []
        for direction in ["N", "S", "E", "W"]:
            for lane in SPAWNING_LANES[direction]:
                for turn_type in LANE_TURN_MAPPING[direction].get(lane, []):
                    self.traffic_lights.append(TrafficLight(direction, lane, turn_type))

        self.spawner = CarsSpawner(max_cars=max_cars)
        self.scheduler = TrafficLightScheduler(self.traffic_lights, self.spawner.cars)
        self.controller = CarsController(self.spawner.cars, self.traffic_lights)

        self.csv_logger = CSVLogger()
        self.cumulative_reward = 0.0

    def _draw_roads(self):
        self.screen.fill((34, 139, 34))

        road_color = (50, 50, 55)
        pygame.draw.rect(self.screen, road_color,
                         (CENTER_X - ROAD_HALF_WIDTH_NS, 0, ROAD_HALF_WIDTH_NS * 2, HEIGHT))
        pygame.draw.rect(self.screen, road_color,
                         (0, CENTER_Y - ROAD_HALF_WIDTH_EW, WIDTH, ROAD_HALF_WIDTH_EW * 2))

        # Draw lane lines for North-South road
        for i in range(LANE_COUNT_NS):
            lane_off = int((i - (LANE_COUNT_NS - 1) / 2) * LANE_WIDTH)
            x_pos = CENTER_X + lane_off
            if abs(x_pos - CENTER_X) > LANE_WIDTH / 2:
                for y in range(0, HEIGHT, 45):
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (x_pos, y), (x_pos, min(HEIGHT, y + 30)), 2)

        # Draw lane lines for East-West road
        for i in range(LANE_COUNT_EW):
            lane_off = int((i - (LANE_COUNT_EW - 1) / 2) * LANE_WIDTH)
            y_pos = CENTER_Y + lane_off
            if abs(y_pos - CENTER_Y) > LANE_WIDTH / 2:
                for x in range(0, WIDTH, 45):
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (x, y_pos), (min(WIDTH, x + 30), y_pos), 2)

        # Road edges
        edge_y_top = CENTER_Y - ROAD_HALF_WIDTH_EW
        edge_y_bottom = CENTER_Y + ROAD_HALF_WIDTH_EW
        edge_x_left = CENTER_X - ROAD_HALF_WIDTH_NS
        edge_x_right = CENTER_X + ROAD_HALF_WIDTH_NS

        pygame.draw.line(self.screen, (255, 220, 0), (0, edge_y_top), (WIDTH, edge_y_top), 3)
        pygame.draw.line(self.screen, (255, 220, 0), (0, edge_y_bottom), (WIDTH, edge_y_bottom), 3)
        pygame.draw.line(self.screen, (255, 220, 0), (edge_x_left, 0), (edge_x_left, HEIGHT), 3)
        pygame.draw.line(self.screen, (255, 220, 0), (edge_x_right, 0), (edge_x_right, HEIGHT), 3)

        # Intersection
        pygame.draw.rect(self.screen, (60, 60, 65),
                         (CENTER_X - ROAD_HALF_WIDTH_NS, CENTER_Y - ROAD_HALF_WIDTH_EW,
                          ROAD_HALF_WIDTH_NS * 2, ROAD_HALF_WIDTH_EW * 2))

        # Draw traffic lights and stop lines
        drawn_lights = set()
        for tl in self.traffic_lights:
            key = (tl.direction, tl.lane)
            if key in drawn_lights:
                continue
            drawn_lights.add(key)

            sx, sy = stop_line_for(tl.direction, tl.lane)

            if tl.direction in ["N", "S"]:
                pygame.draw.line(self.screen, (255, 255, 255),
                                 (int(sx - LANE_WIDTH // 2), int(sy)),
                                 (int(sx + LANE_WIDTH // 2), int(sy)), 3)
            else:
                pygame.draw.line(self.screen, (255, 255, 255),
                                 (int(sx), int(sy - LANE_WIDTH // 2)),
                                 (int(sx), int(sy + LANE_WIDTH // 2)), 3)

            if tl.direction == "N":
                pos = (int(sx), int(sy - 22))
            elif tl.direction == "S":
                pos = (int(sx), int(sy + 22))
            elif tl.direction == "W":
                pos = (int(sx - 22), int(sy))
            else:
                pos = (int(sx + 22), int(sy))

            if tl.state == "GREEN":
                color = (0, 255, 0)
            elif tl.state == "YELLOW":
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)

            pygame.draw.circle(self.screen, color, pos, 8)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 8, 2)

            label = self.small_font.render(f"{tl.direction}{tl.lane}", True, (255, 255, 255))
            self.screen.blit(label, (pos[0] - 12, pos[1] - 14))

    def _draw_hud(self, epoch, decision, epsilon):
        y_offset = 10

        phase_text = "NS" if self.scheduler.current_phase == 0 else "EW"
        state_text = self.scheduler.state

        stats = [
            f"Cars: {len(self.spawner.cars)}/{self.max_cars}",
            f"Phase: {phase_text} | State: {state_text}",
            f"Timer: {self.scheduler.phase_timer:.1f}s",
            f"Waiting - N:{self.cars_waiting['N']} S:{self.cars_waiting['S']} W:{self.cars_waiting['W']} E:{self.cars_waiting['E']}",
            f"Completed: {self.completed_cars}",
            f"Collisions: {self.collision_count}",
            f"Red Violations: {self.controller.red_light_violations}",
            f"Epsilon: {epsilon:.3f}",
            f"Reward: {self.decision_reward:.1f}",
            f"Cumulative: {self.cumulative_reward:.0f}"
        ]

        if epoch:
            stats.append(f"Epoch: {epoch}")
        if decision:
            stats.append(f"Decision: {decision}")

        for i, text in enumerate(stats):
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (10, y_offset + i * 24))

        legend_y = HEIGHT - 190
        legend_title = self.small_font.render("Cleaned Lane Configuration (No N2/S2):", True, (255, 255, 200))
        self.screen.blit(legend_title, (10, legend_y))

        configs = [
            f"Green: {GREEN_DURATION}s | Yellow: {YELLOW_DURATION}s",
            "Phase 0: NS Green, EW Red",
            "Phase 1: EW Green, NS Red",
            "Spawning: N0-1, E2-3, S0-1, W2-3"
        ]
        for i, config in enumerate(configs):
            text = self.small_font.render(config, True, (200, 200, 200))
            self.screen.blit(text, (10, legend_y + 18 + i * 16))

    def update(self, render_game=False, epoch=None, decision=None, epsilon=1.0):
        current_time = time.time()
        dt = min(0.033, current_time - self.last_frame_time)
        self.last_frame_time = current_time

        self.step_count += 1
        self._draw_roads()

        cars_waiting, cars_completed, collisions = self.controller.update_cars_positions(self.screen, render_game)
        self.cars_waiting = cars_waiting
        self.completed_cars += cars_completed
        self.collision_count += len(collisions)

        self.spawner.remove_destroyed_cars(self.controller.lane_queues)
        self.spawner.maybe_spawn_car(self.controller.lane_queues)

        if self.step_count % 200 == 0:
            self.spawner.spawn_agent_car(self.controller.lane_queues)

        self.scheduler.update(dt)

        if render_game:
            self._draw_hud(epoch, decision, epsilon)
            pygame.display.flip()

        self.clock.tick(FPS)

        total_waiting = sum(cars_waiting.values())
        self.last_reward = cars_completed * 10 - total_waiting * 1 - len(collisions) * 30

        self.decision_reward += self.last_reward
        self.epoch_reward += self.last_reward

        return self.last_reward

    def get_state(self):
        state = []
        for d in ["N", "S", "W", "E"]:
            state.append(min(self.cars_waiting[d], 15))
        state.append(self.scheduler.current_phase)
        total_cars = len(self.spawner.cars)
        state.append(min(total_cars // 5, 15))
        state.append(min(int(self.scheduler.phase_timer), 5))
        return tuple(state)

    def compute_reward(self, action: int):
        if action == 0:
            served_waiting = self.cars_waiting["N"] + self.cars_waiting["S"]
            unserved_waiting = self.cars_waiting["W"] + self.cars_waiting["E"]
        else:
            served_waiting = self.cars_waiting["W"] + self.cars_waiting["E"]
            unserved_waiting = self.cars_waiting["N"] + self.cars_waiting["S"]

        reward = served_waiting * 5 - unserved_waiting * 2 - abs(served_waiting - unserved_waiting)
        reward -= self.collision_count * 25
        reward -= (self.controller.red_light_violations * 30 + self.controller.lane_departures * 20)

        if self.repeat_count >= 5:
            reward -= self.repeat_count * 1.5

        return reward

    def set_light_state(self, action: int):
        if self.last_action == action:
            self.repeat_count += 1
        else:
            self.repeat_count = 1
            self.last_action = action

        if action != self.scheduler.current_phase:
            self.scheduler.set_pending_phase(action)

    def reset_decision_reward(self):
        self.decision_reward = 0.0

    def reset_epoch_reward(self):
        self.epoch_reward = 0.0

    def add_to_cumulative(self, reward):
        self.cumulative_reward += reward

    def log_decision(self, step, epoch, decision, action, phase, reward,
                     completed_cars, collisions, total_cars, epsilon):
        self.csv_logger.log(
            step, epoch, decision, action, phase, reward, self.cumulative_reward,
            self.cars_waiting, completed_cars, collisions,
            self.controller.red_light_violations, self.controller.lane_departures,
            total_cars, epsilon
        )

    def reset(self):
        self.last_reward = 0.0
        self.last_action = None
        self.repeat_count = 0
        self.cars_waiting = {"N": 0, "S": 0, "W": 0, "E": 0}
        self.collision_count = 0
        self.step_count = 0
        self.completed_cars = 0
        self.decision_reward = 0.0
        self.epoch_reward = 0.0
        self.last_frame_time = time.time()

        self.spawner = CarsSpawner(max_cars=self.max_cars)
        self.scheduler = TrafficLightScheduler(self.traffic_lights, self.spawner.cars)
        self.controller = CarsController(self.spawner.cars, self.traffic_lights)

    def close_csv(self):
        self.csv_logger.close()

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit()


# =========================
# RL Agent
# =========================
class QLearningAgent:
    def __init__(
            self,
            action_space=2,
            alpha=0.15,
            gamma=0.95,
            epsilon=1.0,
            min_epsilon=0.05,
            epsilon_decay=0.995,
            exploit=False,
    ):
        self.action_space = action_space
        self.q_table = defaultdict(lambda: np.zeros(action_space, dtype=np.float32))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.0 if exploit else epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        max_next_q = float(np.max(self.q_table[next_state]))
        old_q = self.q_table[state][action]
        target = reward + self.gamma * max_next_q
        self.q_table[state][action] = (1 - self.alpha) * old_q + self.alpha * target
        return abs(target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, epochs, path="q_table_cleaned.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "epochs": epochs,
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }, f)
        print(f"[✓] Q-table saved to {path}")

    def load(self, path="q_table_cleaned.pkl"):
        if not os.path.exists(path):
            print(f"[!] No saved Q-table found at {path}")
            return 0
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space, dtype=np.float32),
                                   data.get("q_table", {}))
        self.epsilon = data.get("epsilon", self.epsilon)
        print(f"[✓] Loaded Q-table from {path} (epochs: {data.get('epochs', 0)})")
        return int(data.get("epochs", 0))


# =========================
# Training
# =========================
class TrainingParameters:
    def __init__(self):
        self.agent = QLearningAgent()
        self.environment = TrafficEnv()
        self.total_epochs = 500
        self.decisions_per_epoch = 25


def train(params: TrainingParameters, render_game=False, verbose=False):
    decision_timer = int((GREEN_DURATION + YELLOW_DURATION) * FPS)
    training_losses = []
    episode_rewards = []
    best_reward = float('-inf')

    print("=" * 80)
    print("Starting Training - Cleaned Configuration (No N2/S2)")
    print("=" * 80)
    print(f"CSV Log File: {CSV_FILENAME}")
    print(f"Traffic Light Timing: GREEN={GREEN_DURATION}s, YELLOW={YELLOW_DURATION}s")
    print("Lane Configuration:")
    print("  Spawning: N0-1, E2-3, S0-1, W2-3")
    print("  N0,N1 straight to S4 | S0 straight to N0, S1 left to E4")
    print("  E2,E3 straight to W0 | W2,W3 straight to E0")
    print(f"Max Cars: {params.environment.max_cars}")
    print(f"Learning Rate: {params.agent.alpha}")
    print(f"Discount Factor: {params.agent.gamma}")
    print("=" * 80)

    for epoch in range(params.total_epochs):
        epoch_id = epoch + 1
        decision_count = 1
        params.environment.reset_epoch_reward()

        state = params.environment.get_state()
        action = params.agent.choose_action(state)
        params.environment.set_light_state(action)

        for step in range(decision_timer * params.decisions_per_epoch):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    params.environment.close_csv()
                    params.environment.quit()
                    return

            params.environment.update(
                render_game=render_game,
                epoch=epoch_id,
                decision=decision_count,
                epsilon=params.agent.epsilon
            )

            if step % decision_timer == 0 and step > 0:
                reward = params.environment.compute_reward(action)
                next_state = params.environment.get_state()

                td_error = params.agent.learn(state, action, reward, next_state)
                training_losses.append(td_error)

                params.environment.add_to_cumulative(reward)
                params.environment.log_decision(
                    params.environment.step_count, epoch_id, decision_count, action,
                    params.environment.scheduler.current_phase, reward,
                    params.environment.completed_cars,
                    params.environment.collision_count,
                    len(params.environment.spawner.cars),
                    params.agent.epsilon
                )

                next_action = params.agent.choose_action(next_state)
                params.environment.set_light_state(next_action)
                params.environment.reset_decision_reward()

                decision_count += 1
                state = next_state
                action = next_action

        params.environment.reset()
        params.agent.decay_epsilon()

        avg_reward = params.environment.epoch_reward / params.decisions_per_epoch
        episode_rewards.append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            params.agent.save(epoch_id, "q_table_cleaned_best.pkl")

        if epoch_id % 100 == 0:
            params.agent.save(epoch_id, f"q_table_cleaned_epoch_{epoch_id}.pkl")

        if epoch_id % 10 == 0 or epoch_id == 1:
            recent_loss = np.mean(training_losses[-100:]) if training_losses else 0
            print(f"Epoch {epoch_id:4d}/{params.total_epochs} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Best: {best_reward:8.2f} | "
                  f"ε: {params.agent.epsilon:.4f} | "
                  f"Loss: {recent_loss:.4f} | "
                  f"States: {len(params.agent.q_table)}")

            params.environment.csv_logger.log_epoch_summary(
                epoch_id, avg_reward, best_reward, params.agent.epsilon,
                len(params.agent.q_table), recent_loss
            )

    params.agent.save(params.total_epochs, "q_table_cleaned_complete.pkl")
    params.environment.close_csv()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"CSV saved to: {CSV_FILENAME}")
    print(f"Final epsilon: {params.agent.epsilon:.4f}")
    print(f"Final Q-table size: {len(params.agent.q_table)} states")
    print(f"Best average reward: {best_reward:.2f}")
    print("=" * 80)


def run_inference(render=True):
    print("Loading trained agent...")
    agent = QLearningAgent(exploit=True)
    epochs = agent.load("q_table_cleaned_best.pkl")

    if epochs == 0:
        epochs = agent.load("q_table_cleaned_complete.pkl")

    env = TrafficEnv()
    inference_csv = CSVLogger(f"inference_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    decision_timer = int((GREEN_DURATION + YELLOW_DURATION) * FPS)
    step = 0
    running = True
    clock = pygame.time.Clock()

    total_reward = 0
    decisions = 0
    cumulative_reward = 0

    print("\nStarting inference - Cleaned Configuration (No N2/S2)...")
    print(f"Traffic Lights: {GREEN_DURATION}s Green, {YELLOW_DURATION}s Yellow")
    print("Press ESC or close window to exit")
    print("=" * 80)

    state = env.get_state()
    action = agent.choose_action(state)
    env.set_light_state(action)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if step % decision_timer == 0 and step > 0:
            reward = env.compute_reward(action)
            total_reward += reward
            cumulative_reward += reward
            decisions += 1

            inference_csv.log(
                step, 0, decisions, action, env.scheduler.current_phase, reward, cumulative_reward,
                env.cars_waiting, env.completed_cars, env.collision_count,
                env.controller.red_light_violations, env.controller.lane_departures,
                len(env.spawner.cars), agent.epsilon
            )

            state = env.get_state()
            action = agent.choose_action(state)
            env.set_light_state(action)

            if decisions % 10 == 0:
                avg_reward = total_reward / decisions
                print(f"Decisions: {decisions:4d} | Avg Reward: {avg_reward:7.2f}")

        env.update(render_game=render, epsilon=agent.epsilon)
        step += 1
        clock.tick(FPS)

    inference_csv.close()
    env.quit()

    if decisions > 0:
        print("\n" + "=" * 80)
        print(f"Inference Summary:")
        print(f"  Inference log saved to: {inference_csv.filename}")
        print(f"  Total Decisions: {decisions}")
        print(f"  Average Reward: {total_reward / decisions:.2f}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Intersection RL - Cleaned Configuration")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="inference")
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--max-cars", type=int, default=50)

    args = parser.parse_args()

    if args.mode == "train":
        params = TrainingParameters()
        params.total_epochs = args.epochs
        params.environment.max_cars = args.max_cars
        train(params, render_game=args.render, verbose=args.verbose)
    else:
        run_inference(render=args.render)
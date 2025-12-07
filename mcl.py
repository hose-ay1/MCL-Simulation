import pygame
import math
import random
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

# PyGame initialization
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MCL Robot Localization Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)
button_font = pygame.font.SysFont('Arial', 18)

# Colors
BACKGROUND = (25, 25, 35)
CUSTOM_BG = (40, 40, 50)
PANEL_BG = (50, 50, 65)
PANEL_BORDER = (80, 80, 100)
FIELD_BG = (240, 240, 235)
WALL_COLOR = (60, 60, 80)
OBSTACLE_COLOR = (200, 100, 100)
ROBOT_COLOR = (50, 150, 220, 100)  # Semi-transparent
ROBOT_OUTLINE = (50, 150, 220, 200)
SENSOR_COLOR = (240, 200, 60)
SENSOR_OUT_RANGE = (255, 100, 100)
RAY_COLOR = (255, 150, 50, 180)
RAY_OUT_RANGE = (255, 80, 80, 180)
PARTICLE_LOW = (255, 50, 50)
PARTICLE_MID = (255, 180, 50)
PARTICLE_HIGH = (50, 255, 100)
ESTIMATE_COLOR = (200, 80, 240)
TEXT_COLOR = (230, 230, 230)
TEXT_DISABLED = (150, 150, 150)
SLIDER_BG = (70, 70, 80)
SLIDER_FG = (90, 180, 240)
BUTTON_COLOR = (60, 160, 80)
BUTTON_HOVER = (80, 200, 100)
BUTTON_DISABLED = (100, 100, 100)
BUTTON_TEXT = (255, 255, 255)
HIGHLIGHT = (255, 255, 100)
GRID_COLOR = (100, 100, 120, 50)
KIDNAP_COLOR = (220, 100, 100)
KIDNAP_ACTIVE_COLOR = (255, 50, 50)
HELP_COLOR = (100, 100, 220)
SCROLLBAR_COLOR = (80, 80, 100)
SCROLLBAR_HANDLE = (120, 120, 140)

# Field parameters
FIELD_WIDTH = 144  # inches
FIELD_HEIGHT = 144
SCALE = 4.5  # pixels per inch
FIELD_OFFSET_X = 350
FIELD_OFFSET_Y = 100

# Sensor parameters
SENSOR_MAX_RANGE = 78.74016  # inches (2 meters)
SENSOR_MIN_RANGE = 0.0  # inches
WALL_THICKNESS = 2.5  # inches

@dataclass
class Rectangle:
    """Axis-aligned rectangle obstacle"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class FieldMap:
    """Represents the field with walls and obstacles"""
    def __init__(self):
        self.width = FIELD_WIDTH
        self.height = FIELD_HEIGHT
        
        # Create walls with 1.27" thickness towards the inside
        self.walls = [
            Rectangle(-self.width/2, -self.height/2, 
                     self.width/2, -self.height/2 + WALL_THICKNESS),
            Rectangle(-self.width/2, self.height/2 - WALL_THICKNESS, 
                     self.width/2, self.height/2),
            Rectangle(-self.width/2, -self.height/2,
                     -self.width/2 + WALL_THICKNESS, self.height/2),
            Rectangle(self.width/2 - WALL_THICKNESS, -self.height/2,
                     self.width/2, self.height/2),
        ]
        self.obstacles: List[Rectangle] = []
        self.create_quadrant_object()
        
        # Try to load background image
        self.background_image = None
        self.load_background()
    
    def load_background(self):
        """Load field background image"""
        try:
            image_paths = [
                "field.png",
                "./field.png",
                os.path.join(os.path.dirname(__file__), "field.png")
            ]
            
            for path in image_paths:
                if os.path.exists(path):
                    self.background_image = pygame.image.load(path)
                    target_width = int(FIELD_WIDTH * SCALE)
                    target_height = int(FIELD_HEIGHT * SCALE)
                    self.background_image = pygame.transform.scale(
                        self.background_image, (target_width, target_height)
                    )
                    break
        except Exception as e:
            print(f"Could not load background image: {e}")
            self.background_image = None
    
    def create_quadrant_object(self):
        """Create the object in all 4 quadrants"""
        lines_q1 = [
            [(23.5, 44.5), (21, 47)],
            [(21, 47), (23.5, 49.5)]
        ]
        
        quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        
        for x_sign, y_sign in quadrants:
            for line in lines_q1:
                x1, y1 = line[0][0] * x_sign, line[0][1] * y_sign
                x2, y2 = line[1][0] * x_sign, line[1][1] * y_sign
                self.add_line_as_obstacle(x1, y1, x2, y2, thickness=0.5)
    
    def add_line_as_obstacle(self, x1: float, y1: float, x2: float, y2: float, thickness: float = 0.5):
        """Add a line as a thin rectangle obstacle"""
        min_x = min(x1, x2) - thickness/2
        max_x = max(x1, x2) + thickness/2
        min_y = min(y1, y2) - thickness/2
        max_y = max(y1, y2) + thickness/2
        self.obstacles.append(Rectangle(min_x, min_y, max_x, max_y))
    
    def raycast(self, origin_x: float, origin_y: float, angle: float, max_range: float = SENSOR_MAX_RANGE) -> Tuple[float, bool]:
        """Cast a ray and return distance to nearest intersection and if it's in range"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        eps = 0.001
        
        min_distance = max_range
        
        # Check all rectangles (walls and obstacles)
        for rect in self.walls + self.obstacles:
            t1 = (rect.x_min - origin_x) / cos_a if abs(cos_a) > eps else float('inf')
            t2 = (rect.x_max - origin_x) / cos_a if abs(cos_a) > eps else float('inf')
            t3 = (rect.y_min - origin_y) / sin_a if abs(sin_a) > eps else float('inf')
            t4 = (rect.y_max - origin_y) / sin_a if abs(sin_a) > eps else float('inf')
            
            tmin = max(min(t1, t2), min(t3, t4))
            tmax = min(max(t1, t2), max(t3, t4))
            
            if tmax >= 0 and tmin <= tmax:
                distance = max(tmin, 0)
                if 0 < distance < min_distance:
                    min_distance = distance
        
        in_range = min_distance < max_range - 0.1
        return min_distance, in_range

class Sensor:
    """Distance sensor on the robot"""
    def __init__(self, offset_x: float, offset_y: float, angle_offset: float):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.angle_offset = angle_offset  # 0° is forward, positive is counter-clockwise
        self.distance = 0.0
        self.max_range = SENSOR_MAX_RANGE
        self.in_range = True
        self.cut_out = False  # True if beam is longer than max range
    
    def get_global_position(self, robot_x: float, robot_y: float, robot_theta: float) -> Tuple[float, float]:
        """Get sensor position in global coordinates"""
        # Rotate sensor offset by robot orientation
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)
        
        # Apply rotation to sensor offset
        sensor_x = robot_x + self.offset_x * cos_theta - self.offset_y * sin_theta
        sensor_y = robot_y + self.offset_x * sin_theta + self.offset_y * cos_theta
        
        return sensor_x, sensor_y
    
    def get_global_angle(self, robot_theta: float) -> float:
        """Get sensor angle in global coordinates"""
        return robot_theta + self.angle_offset
    
    def update(self, robot_x: float, robot_y: float, robot_theta: float, field_map: FieldMap):
        """Update sensor reading"""
        sensor_x, sensor_y = self.get_global_position(robot_x, robot_y, robot_theta)
        sensor_angle = self.get_global_angle(robot_theta)
        
        measured_distance, self.in_range = field_map.raycast(sensor_x, sensor_y, sensor_angle, self.max_range)
        self.cut_out = not self.in_range
        
        # Add Gaussian noise to simulate real sensor (only if in range)
        if self.in_range:
            self.distance = measured_distance + random.gauss(0, 0.5)
            self.distance = max(SENSOR_MIN_RANGE, min(self.distance, self.max_range))
        else:
            self.distance = self.max_range

class Robot:
    """Robot with configurable dimensions and sensors"""
    def __init__(self, width: float = 12.5, height: float = 15.0):
        self.width = max(12, min(18, width))
        self.height = max(12, min(18, height))
        self.x = 0
        self.y = 0
        self.theta = 0  # 0° is right, positive is counter-clockwise
        self.speed = 0.0
        self.angular_speed = 0.0
        self.max_speed = 40.0
        self.max_angular_speed = 3.0
        self.sensors: List[Sensor] = []
        self.sensor_positions: List[Tuple[float, float, float]] = []
        
        # Store previous position for delta calculation
        self.prev_x = 0
        self.prev_y = 0
        self.prev_theta = 0
    
    def set_size(self, width: float, height: float):
        """Set robot size with clamping"""
        self.width = max(12, min(18, width))
        self.height = max(12, min(18, height))
    
    def add_sensor(self, offset_x: float, offset_y: float, angle_offset: float):
        """Add a sensor at specified position relative to robot center"""
        max_offset_x = self.height / 2 + 2
        max_offset_y = self.width / 2 + 2
        offset_x = max(-max_offset_x, min(max_offset_x, offset_x))
        offset_y = max(-max_offset_y, min(max_offset_y, offset_y))
        
        self.sensor_positions.append((offset_x, offset_y, angle_offset))
        self.sensors.append(Sensor(offset_x, offset_y, angle_offset))
    
    def clear_sensors(self):
        """Remove all sensors"""
        self.sensors = []
        self.sensor_positions = []
    
    def update(self, dt: float):
        """Update robot position based on speed and store previous position"""
        # Store previous position before updating
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
        
        # Calculate new position
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        self.theta += self.angular_speed * dt
        self.theta %= (2 * math.pi)
    
    def get_deltas(self) -> Tuple[float, float, float]:
        """Get delta x, y, theta from previous position"""
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        dtheta = self.theta - self.prev_theta
        
        # Normalize angle difference to [-pi, pi]
        while dtheta > math.pi:
            dtheta -= 2 * math.pi
        while dtheta < -math.pi:
            dtheta += 2 * math.pi
            
        return dx, dy, dtheta
    
    def update_sensors(self, field_map: FieldMap):
        """Update all sensor readings"""
        for sensor in self.sensors:
            sensor.update(self.x, self.y, self.theta, field_map)
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """Get robot corners for drawing"""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        half_w = self.width / 2
        half_h = self.height / 2
        
        # Define corners relative to robot center
        # Note: height is along forward/backward, width is left/right
        corners_local = [
            ( half_h,  half_w),  # Front-right
            ( half_h, -half_w),  # Front-left
            (-half_h, -half_w),  # Back-left
            (-half_h,  half_w)   # Back-right
        ]
        
        corners_global = []
        for x_local, y_local in corners_local:
            # Rotate corner by robot orientation
            x_global = self.x + x_local * cos_theta - y_local * sin_theta
            y_global = self.y + x_local * sin_theta + y_local * cos_theta
            corners_global.append((x_global, y_global))
        
        return corners_global

class Particle:
    """Represents a single hypothesis of robot pose"""
    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta % (2 * math.pi)
        self.weight = weight
        self.alive = True
    
    def is_in_field(self) -> bool:
        """Check if particle is within field bounds"""
        return (-72 < self.x < 72) and (-72 < self.y < 72)
    
    def predict(self, delta_x: float, delta_y: float, delta_theta: float,
                motion_noise: Tuple[float, float, float] = (0.1, 0.1, 0.01)):
        """Move particle with noise using absolute position update"""
        # Apply motion
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Add noise
        self.x += random.gauss(0, motion_noise[0])
        self.y += random.gauss(0, motion_noise[1])
        self.theta += random.gauss(0, motion_noise[2])
        self.theta %= (2 * math.pi)
        
        # Kill particle if it goes outside field
        self.alive = self.is_in_field()
    
    def measurement_probability(self, sensors: List[Sensor], field_map: FieldMap, gaussian_stdev: float) -> float:
        """Calculate particle weight based on sensor measurements"""
        if not sensors or not self.alive:
            return 0.0
        
        total_prob = 1.0
        sensor_count = 0
        
        for sensor in sensors:
            # Get sensor position and angle for this particle
            sensor_x, sensor_y = sensor.get_global_position(self.x, self.y, self.theta)
            sensor_angle = sensor.get_global_angle(self.theta)
            
            # Get expected distance via raycast
            expected_dist, expected_in_range = field_map.raycast(sensor_x, sensor_y, sensor_angle, sensor.max_range)
            
            # Skip if sensor is out of range and particle also expects out of range
            if sensor.cut_out and not expected_in_range:
                # Both agree on no obstacle - low but non-zero probability
                prob = 0.1
            elif not sensor.cut_out and expected_in_range:
                # Both have valid readings - Gaussian probability
                error = abs(sensor.distance - expected_dist)
                # Clamp error to avoid extreme values
                error = min(error, 3 * gaussian_stdev)
                prob = math.exp(-0.5 * (error ** 2) / (gaussian_stdev ** 2))
            else:
                # Disagreement - very low probability
                prob = 0.001
            
            total_prob *= prob
            sensor_count += 1
        
        if sensor_count == 0:
            return 0.001
        
        # Geometric mean
        return total_prob ** (1.0 / sensor_count)
    
    def clone(self) -> 'Particle':
        return Particle(self.x, self.y, self.theta, self.weight)

class MCL:
    """Monte Carlo Localization with kidnap recovery"""
    n_particles = 1000
    gaussian_stdev = 1.5
    gaussian_factor = 1.0
    
    def __init__(self, field_map: FieldMap, robot_x: float = 0, robot_y: float = 0, robot_theta: float = 0):
        self.field_map = field_map
        self.particles: List[Particle] = []
        self.average_pose = (0.0, 0.0, 0.0)
        self.motion_noise_xy = 0.15  # Separate XY and theta noise
        self.motion_noise_theta = 0.015
        self.kidnapped = False
        self.kidnap_counter = 0
        self.last_good_estimate = (robot_x, robot_y, robot_theta)
        self.initialize_around_pose(robot_x, robot_y, robot_theta, 5.0)
    
    def detect_kidnap(self, sensors: List[Sensor]) -> bool:
        """Detect if robot has been kidnapped based on sensor consistency"""
        if not sensors:
            return False
        
        # Count how many particles agree with sensor readings
        good_particles = 0
        for particle in self.particles:
            if particle.alive:
                # Quick check: does at least one sensor reading make sense for this particle?
                for sensor in sensors:
                    if sensor.in_range:
                        sensor_x, sensor_y = sensor.get_global_position(particle.x, particle.y, particle.theta)
                        sensor_angle = sensor.get_global_angle(particle.theta)
                        expected_dist, in_range = self.field_map.raycast(sensor_x, sensor_y, sensor_angle, sensor.max_range)
                        if in_range and abs(sensor.distance - expected_dist) < 20:  # Within 20 inches
                            good_particles += 1
                            break
        
        # If less than 10% of particles make sense, we're probably kidnapped
        kidnap_detected = (good_particles / max(1, len(self.particles))) < 0.1
        
        if kidnap_detected:
            self.kidnap_counter += 1
        else:
            self.kidnap_counter = max(0, self.kidnap_counter - 1)
        
        return self.kidnap_counter > 3  # Need multiple consecutive detections
    
    def inject_random_particles(self, n: int = 100):
        """Inject random particles to help recover from kidnapping"""
        new_particles = []
        for _ in range(n):
            # Random position in field
            x = random.uniform(-self.field_map.width/2 + 5, self.field_map.width/2 - 5)
            y = random.uniform(-self.field_map.height/2 + 5, self.field_map.height/2 - 5)
            theta = random.uniform(0, 2 * math.pi)
            particle = Particle(x, y, theta)
            if particle.is_in_field():
                new_particles.append(particle)
        
        # Add to existing particles, removing some low-weight ones
        if len(self.particles) > n:
            # Remove n lowest-weight particles
            self.particles.sort(key=lambda p: p.weight)
            self.particles = self.particles[n:] + new_particles
        else:
            self.particles = self.particles + new_particles
    
    def adapt_noise_for_kidnap(self):
        """Increase motion noise to help exploration when kidnapped"""
        if self.kidnapped:
            # Double the noise when kidnapped
            return (self.motion_noise_xy * 2, 
                    self.motion_noise_xy * 2, 
                    self.motion_noise_theta * 2)
        return (self.motion_noise_xy, self.motion_noise_xy, self.motion_noise_theta)
    
    def initialize_uniform(self):
        """Initialize particles uniformly across the field"""
        self.particles = []
        for _ in range(self.n_particles):
            x = random.uniform(-self.field_map.width/2 + 5, self.field_map.width/2 - 5)
            y = random.uniform(-self.field_map.height/2 + 5, self.field_map.height/2 - 5)
            theta = random.uniform(0, 2 * math.pi)
            self.particles.append(Particle(x, y, theta))
    
    def initialize_around_pose(self, x: float, y: float, theta: float, variance: float = 5.0):
        """Initialize particles around a specific pose"""
        self.particles = []
        for _ in range(self.n_particles):
            px = x + random.gauss(0, variance)
            py = y + random.gauss(0, variance)
            ptheta = theta + random.gauss(0, 0.1)
            particle = Particle(px, py, ptheta)
            if particle.is_in_field():
                self.particles.append(particle)
        
        while len(self.particles) < self.n_particles:
            px = x + random.gauss(0, variance)
            py = y + random.gauss(0, variance)
            ptheta = theta + random.gauss(0, 0.1)
            particle = Particle(px, py, ptheta)
            if particle.is_in_field():
                self.particles.append(particle)
    
    def update_step(self, delta_x: float, delta_y: float, delta_theta: float):
        """Prediction step with kidnap recovery"""
        current_noise = self.adapt_noise_for_kidnap()
        
        for particle in self.particles:
            particle.predict(delta_x, delta_y, delta_theta, current_noise)
        
        # Remove dead particles (outside field)
        self.particles = [p for p in self.particles if p.alive]
        
        # If kidnapped or very few particles, inject random ones
        particles_lost = len(self.particles) < self.n_particles * 0.3
        if self.kidnapped or particles_lost:
            particles_to_inject = max(50, int(self.n_particles * 0.1))
            self.inject_random_particles(particles_to_inject)
        
        # Always maintain particle count
        while len(self.particles) < self.n_particles:
            # Clone a random survivor
            if self.particles:
                random_particle = random.choice(self.particles)
                self.particles.append(random_particle.clone())
            else:
                # No survivors, create random particle
                x = random.uniform(-self.field_map.width/2 + 5, self.field_map.width/2 - 5)
                y = random.uniform(-self.field_map.height/2 + 5, self.field_map.height/2 - 5)
                theta = random.uniform(0, 2 * math.pi)
                self.particles.append(Particle(x, y, theta))
    
    def resample_step(self, sensors: List[Sensor]):
        """Update and resample particles with kidnap detection"""
        # First, check if we've been kidnapped
        if sensors:
            self.kidnapped = self.detect_kidnap(sensors)
        
        # Calculate weights
        total_weight = 0.0
        for particle in self.particles:
            particle.weight = particle.measurement_probability(sensors, self.field_map, self.gaussian_stdev) * self.gaussian_factor
            total_weight += particle.weight
        
        if total_weight <= 0 or self.kidnapped:
            # If kidnapped or all weights are zero, do global exploration
            if self.kidnapped:
                print("Kidnap detected! Injecting exploration particles...")
                # Keep only top 20% of particles
                self.particles.sort(key=lambda p: p.weight, reverse=True)
                keep_count = max(50, int(len(self.particles) * 0.2))
                self.particles = self.particles[:keep_count]
                
                # Add exploration particles
                exploration_count = self.n_particles - len(self.particles)
                self.inject_random_particles(exploration_count)
            
            # Recalculate weights
            total_weight = 0.0
            for particle in self.particles:
                particle.weight = particle.measurement_probability(sensors, self.field_map, self.gaussian_stdev) * self.gaussian_factor
                total_weight += particle.weight
            
            if total_weight <= 0:
                # Still no good weights, use uniform
                for particle in self.particles:
                    particle.weight = 1.0 / len(self.particles)
                total_weight = 1.0
        
        # Normalize weights
        for particle in self.particles:
            particle.weight /= total_weight
        
        # Effective sample size check
        eff_n = 1.0 / sum(p.weight ** 2 for p in self.particles)
        
        # Resample if effective sample size is too low OR if kidnapped
        if eff_n < self.n_particles * 0.5 or self.kidnapped:
            # Low variance resampling
            new_particles = []
            r = random.uniform(0, 1 / len(self.particles))
            c = self.particles[0].weight
            i = 0
            
            for m in range(self.n_particles):
                u = r + m / self.n_particles
                while u > c and i < len(self.particles) - 1:
                    i += 1
                    c += self.particles[i].weight
                if i < len(self.particles):
                    new_particles.append(self.particles[i].clone())
            
            self.particles = new_particles
        
        # Update average pose
        self.update_average_pose()
        
        # If we have a good estimate, reset kidnap flag
        if not self.kidnapped:
            self.last_good_estimate = self.average_pose
            self.kidnap_counter = 0
    
    def update_average_pose(self):
        """Calculate weighted average pose"""
        if not self.particles:
            return
        
        alive_particles = [p for p in self.particles if p.alive]
        if not alive_particles:
            return
        
        # Weighted average of positions
        total_weight = sum(p.weight for p in alive_particles)
        if total_weight == 0:
            return
        
        x = sum(p.x * p.weight for p in alive_particles) / total_weight
        y = sum(p.y * p.weight for p in alive_particles) / total_weight
        
        # Circular mean for angles
        sin_sum = sum(math.sin(p.theta) * p.weight for p in alive_particles)
        cos_sum = sum(math.cos(p.theta) * p.weight for p in alive_particles)
        theta = math.atan2(sin_sum, cos_sum)
        
        self.average_pose = (x, y, theta)
    
    def run(self, sensors: List[Sensor], delta_x: float, delta_y: float, delta_theta: float):
        """Complete MCL step with kidnap recovery"""
        self.update_step(delta_x, delta_y, delta_theta)
        self.resample_step(sensors)
        return self.average_pose

class Slider:
    """UI Slider for parameter adjustment"""
    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float, 
                 initial_val: float, label: str, value_format: str = ":.2f"):
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.value_format = value_format
        self.dragging = False
    
    def draw(self, screen):
        if ":d" in self.value_format:
            value_text = f"{int(self.value)}"
        elif ":.1f" in self.value_format:
            value_text = f"{self.value:.1f}"
        elif ":.2f" in self.value_format:
            value_text = f"{self.value:.2f}"
        elif ":.3f" in self.value_format:
            value_text = f"{self.value:.3f}"
        else:
            value_text = f"{self.value:.2f}"
            
        label_text = font.render(f"{self.label}: {value_text}", True, TEXT_COLOR)
        screen.blit(label_text, (self.x, self.y - 20))
        
        pygame.draw.rect(screen, SLIDER_BG, (self.x, self.y, self.width, self.height), border_radius=3)
        
        handle_x = self.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.width
        pygame.draw.circle(screen, SLIDER_FG, (int(handle_x), self.y + self.height//2), 8)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if (self.x <= mouse_x <= self.x + self.width and 
                self.y <= mouse_y <= self.y + self.height):
                self.dragging = True
                self.update_value(mouse_x)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(event.pos[0])
    
    def update_value(self, mouse_x):
        relative_x = max(0, min(self.width, mouse_x - self.x))
        self.value = self.min_val + (relative_x / self.width) * (self.max_val - self.min_val)
        return self.value

class Button:
    """UI Button"""
    def __init__(self, x: int, y: int, width: int, height: int, label: str, 
                 color: Tuple[int, int, int] = BUTTON_COLOR, text_color: Tuple[int, int, int] = BUTTON_TEXT):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color
        self.hover_color = tuple(min(255, c + 30) for c in color)
        self.text_color = text_color
        self.hovered = False
        self.enabled = True
    
    def draw(self, screen):
        color = self.hover_color if self.hovered and self.enabled else self.color
        if not self.enabled:
            color = BUTTON_DISABLED
        
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height), border_radius=5)
        pygame.draw.rect(screen, PANEL_BORDER, (self.x, self.y, self.width, self.height), 2, border_radius=5)
        
        text = button_font.render(self.label, True, self.text_color if self.enabled else TEXT_DISABLED)
        text_rect = text.get_rect(center=(self.x + self.width//2, self.y + self.height//2))
        screen.blit(text, text_rect)
    
    def handle_event(self, event):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.hovered = (self.x <= mouse_x <= self.x + self.width and 
                       self.y <= mouse_y <= self.y + self.height)
        
        if event.type == pygame.MOUSEBUTTONDOWN and self.hovered and self.enabled:
            return True
        return False

class RobotCustomizationScreen:
    """Screen for customizing the robot and MCL parameters"""
    def __init__(self):
        self.robot = Robot()
        self.dragging_sensor = None
        self.selected_sensor = None
        self.show_help = False
        
        # Scroll variables for help popup
        self.help_scroll_y = 0
        self.scroll_dragging = False
        self.scroll_start_y = 0
        self.help_content_height = 0
        
        self.panels = {
            "robot": pygame.Rect(50, 100, 350, 620),
            "params": pygame.Rect(420, 100, 350, 620),
            "buttons": pygame.Rect(790, 100, 350, 620)
        }
        
        # Robot configuration sliders (moved down by 10 pixels)
        self.width_slider = Slider(70, 160, 250, 12, 18, 12.5, "Robot Width", ":.1f")
        self.height_slider = Slider(70, 210, 250, 12, 18, 15.0, "Robot Length", ":.1f")
        self.speed_slider = Slider(70, 260, 250, 20.0, 60.0, 40.0, "Max Speed", ":.1f")
        self.angular_speed_slider = Slider(70, 310, 250, 0.5, 10.0, 3.0, "Turn Speed", ":.1f")
        
        # MCL parameter sliders (moved down by 10 pixels)
        self.particles_slider = Slider(440, 160, 250, 100, 3000, 1000, "Number of Particles", ":d")
        self.stdev_slider = Slider(440, 210, 250, 0.1, 5.0, 1.5, "Gaussian StDev", ":.1f")
        self.factor_slider = Slider(440, 260, 250, 0.1, 5.0, 1.0, "Gaussian Factor", ":.1f")
        self.motion_noise_xy_slider = Slider(440, 310, 250, 0.01, 0.5, 0.15, "Motion Noise XY", ":.2f")
        self.motion_noise_theta_slider = Slider(440, 360, 250, 0.001, 0.1, 0.015, "Motion Noise Theta", ":.3f")
        
        # Buttons
        self.play_button = Button(810, 200, 300, 60, "PLAY SIMULATION", (80, 180, 90))
        self.reset_sensors_button = Button(810, 280, 300, 50, "CLEAR ALL SENSORS", (180, 100, 80))
        self.default_sensors_button = Button(810, 350, 300, 50, "SET DEFAULT SENSORS", (100, 100, 180))
        self.help_button = Button(810, 420, 300, 50, "CONTROLS & INSTRUCTIONS", HELP_COLOR)
        
        # Set default sensors (0° is right, positive is CCW)
        self.set_default_sensors()
    
    def handle_event(self, event):
        if self.show_help:
            # Handle events for the help popup
            popup_width = 700
            popup_height = 500
            popup_x = (WIDTH - popup_width) // 2
            popup_y = (HEIGHT - popup_height) // 2
            
            # Check if click is inside popup
            mouse_x, mouse_y = pygame.mouse.get_pos()
            popup_clicked = (popup_x <= mouse_x <= popup_x + popup_width and 
                            popup_y <= mouse_y <= popup_y + popup_height)
            
            # Handle mouse wheel for scrolling
            if event.type == pygame.MOUSEWHEEL:
                if popup_clicked:
                    self.help_scroll_y -= event.y * 20  # Scroll speed
                    self.help_scroll_y = max(-self.help_content_height + 400, min(0, self.help_scroll_y))
            
            # Handle scrollbar dragging
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicking on scrollbar
                scrollbar_x = popup_x + popup_width - 20
                scrollbar_width = 10
                
                # Calculate scrollbar position
                content_height = self.help_content_height
                visible_height = 400
                scroll_ratio = -self.help_scroll_y / max(1, content_height - visible_height)
                scrollbar_height = max(30, visible_height * visible_height / max(1, content_height))
                scrollbar_top = popup_y + 80 + scroll_ratio * (visible_height - scrollbar_height)
                
                if (scrollbar_x <= mouse_x <= scrollbar_x + scrollbar_width and 
                    scrollbar_top <= mouse_y <= scrollbar_top + scrollbar_height):
                    self.scroll_dragging = True
                    self.scroll_start_y = mouse_y
                
                # Check if clicking close button
                close_button_rect = pygame.Rect(popup_x + popup_width - 40, popup_y + 10, 30, 30)
                if close_button_rect.collidepoint(mouse_x, mouse_y):
                    self.show_help = False
                    self.help_scroll_y = 0
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.scroll_dragging = False
            
            elif event.type == pygame.MOUSEMOTION and self.scroll_dragging:
                dy = mouse_y - self.scroll_start_y
                visible_height = 400
                content_height = self.help_content_height
                
                if content_height > visible_height:
                    scroll_ratio = dy / (visible_height - max(30, visible_height * visible_height / content_height))
                    self.help_scroll_y -= scroll_ratio * (content_height - visible_height)
                    self.help_scroll_y = max(-content_height + visible_height, min(0, self.help_scroll_y))
                    self.scroll_start_y = mouse_y
            
            # Close on ESC
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.show_help = False
                self.help_scroll_y = 0
            
            # Don't process other events when help is showing
            return None
        
        # Normal event handling when help is not showing
        self.width_slider.handle_event(event)
        self.height_slider.handle_event(event)
        self.speed_slider.handle_event(event)
        self.angular_speed_slider.handle_event(event)
        self.particles_slider.handle_event(event)
        self.stdev_slider.handle_event(event)
        self.factor_slider.handle_event(event)
        self.motion_noise_xy_slider.handle_event(event)
        self.motion_noise_theta_slider.handle_event(event)
        
        if self.play_button.handle_event(event):
            return "play"
        
        if self.reset_sensors_button.handle_event(event):
            self.robot.clear_sensors()
            self.selected_sensor = None
        
        if self.default_sensors_button.handle_event(event):
            self.set_default_sensors()
        
        if self.help_button.handle_event(event):
            self.show_help = True
            self.help_scroll_y = 0
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            if self.panels["robot"].collidepoint(mouse_x, mouse_y):
                center_x = self.panels["robot"].centerx
                center_y = 450
                
                if center_y - 100 < mouse_y < center_y + 100:
                    offset_x = (mouse_x - center_x) / 8
                    offset_y = (mouse_y - center_y) / 8
                    
                    self.robot.add_sensor(offset_x, offset_y, 0)
                    self.selected_sensor = len(self.robot.sensors) - 1
            
            for i, sensor in enumerate(self.robot.sensors):
                center_x = self.panels["robot"].centerx
                center_y = 450
                
                sensor_x = center_x + sensor.offset_x * 8
                sensor_y = center_y + sensor.offset_y * 8
                
                if math.sqrt((mouse_x - sensor_x)**2 + (mouse_y - sensor_y)**2) < 10:
                    # Create a new sensor at the clicked position
                    offset_x = (mouse_x - center_x) / 8
                    offset_y = (mouse_y - center_y) / 8
                    self.robot.add_sensor(offset_x, offset_y, 0)
                    self.selected_sensor = len(self.robot.sensors) - 1
                    self.dragging_sensor = self.selected_sensor
                    break
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_sensor = None
        
        elif event.type == pygame.MOUSEMOTION and self.dragging_sensor is not None:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            center_x = self.panels["robot"].centerx
            center_y = 450
            
            offset_x = (mouse_x - center_x) / 8
            offset_y = (mouse_y - center_y) / 8
            
            max_offset_x = self.robot.height / 2
            max_offset_y = self.robot.width / 2
            offset_x = max(-max_offset_x, min(max_offset_x, offset_x))
            offset_y = max(-max_offset_y, min(max_offset_y, offset_y))
            
            self.robot.sensors[self.dragging_sensor].offset_x = offset_x
            self.robot.sensors[self.dragging_sensor].offset_y = offset_y
            self.robot.sensor_positions[self.dragging_sensor] = (offset_x, offset_y, 
                self.robot.sensors[self.dragging_sensor].angle_offset)
        
        elif event.type == pygame.KEYDOWN:
            if self.selected_sensor is not None:
                if event.key == pygame.K_LEFT:
                    self.robot.sensors[self.selected_sensor].angle_offset -= math.pi/8
                elif event.key == pygame.K_RIGHT:
                    self.robot.sensors[self.selected_sensor].angle_offset += math.pi/8
                elif event.key in [pygame.K_DELETE, pygame.K_BACKSPACE]:
                    self.robot.sensors.pop(self.selected_sensor)
                    self.robot.sensor_positions.pop(self.selected_sensor)
                    self.selected_sensor = None
        
        return None
    
    def set_default_sensors(self):
        """Set default sensor configuration (0° is right, positive is CCW)"""
        self.robot.clear_sensors()
        # Front center - facing forward (0°)
        self.robot.add_sensor(self.robot.height/2, 0, 0)
        # Right side - facing right (-90° from front)
        self.robot.add_sensor(0, -self.robot.width/2, -math.pi/2)
        # Back center - facing backward (180°)
        self.robot.add_sensor(-self.robot.height/2, 0, math.pi)
        # Left side - facing left (90° from front)
        self.robot.add_sensor(0, self.robot.width/2, math.pi/2)
        self.selected_sensor = 0
    
    def update(self):
        self.robot.set_size(self.width_slider.value, self.height_slider.value)
        self.robot.max_speed = self.speed_slider.value
        self.robot.max_angular_speed = self.angular_speed_slider.value
        
        MCL.n_particles = int(self.particles_slider.value)
        MCL.gaussian_stdev = self.stdev_slider.value
        MCL.gaussian_factor = self.factor_slider.value
        # Store noise values as class variables for use in FieldScreen
        MCL.motion_noise_xy_default = self.motion_noise_xy_slider.value
        MCL.motion_noise_theta_default = self.motion_noise_theta_slider.value
    
    def draw(self, screen):
        screen.fill(CUSTOM_BG)
        
        title = title_font.render("ROBOT CUSTOMIZATION", True, TEXT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))
        
        for panel_name, panel_rect in self.panels.items():
            pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
            pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=10)
            
            if panel_name == "robot":
                text = font.render("ROBOT CONFIGURATION", True, TEXT_COLOR)
            elif panel_name == "params":
                text = font.render("MCL PARAMETERS", True, TEXT_COLOR)
            else:
                text = font.render("SIMULATION CONTROLS", True, TEXT_COLOR)
            
            screen.blit(text, (panel_rect.x + 20, panel_rect.y + 15))
        
        # Draw robot visualization
        center_x, center_y = self.panels["robot"].centerx, 450
        scale = 8
        
        robot_rect = pygame.Rect(
            center_x - self.robot.height/2 * scale,
            center_y - self.robot.width/2 * scale,
            self.robot.height * scale,
            self.robot.width * scale
        )
        
        robot_surface = pygame.Surface((self.robot.height * scale, self.robot.width * scale), pygame.SRCALPHA)
        pygame.draw.rect(robot_surface, ROBOT_COLOR, robot_surface.get_rect(), border_radius=5)
        pygame.draw.rect(robot_surface, ROBOT_OUTLINE, robot_surface.get_rect(), 2, border_radius=5)
        screen.blit(robot_surface, (center_x - self.robot.height/2 * scale, center_y - self.robot.width/2 * scale))
        
        # Draw heading arrow (0° is right)
        heading_length = 15
        heading_x = center_x + heading_length
        heading_y = center_y
        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (heading_x, heading_y), 3)
        pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 4)
        
        # Draw sensors
        for i, sensor in enumerate(self.robot.sensors):
            sensor_x = center_x + sensor.offset_x * scale
            sensor_y = center_y + sensor.offset_y * scale
            
            color = HIGHLIGHT if i == self.selected_sensor else SENSOR_COLOR
            pygame.draw.circle(screen, color, (int(sensor_x), int(sensor_y)), 6)
            
            dir_length = 12
            dir_x = sensor_x + dir_length * math.cos(sensor.angle_offset)
            dir_y = sensor_y + dir_length * math.sin(sensor.angle_offset)
            pygame.draw.line(screen, color, (sensor_x, sensor_y), (dir_x, dir_y), 2)
            
            num_text = font.render(str(i+1), True, (0, 0, 0))
            screen.blit(num_text, (sensor_x - 4, sensor_y - 6))
        
        # Draw sliders
        self.width_slider.draw(screen)
        self.height_slider.draw(screen)
        self.speed_slider.draw(screen)
        self.angular_speed_slider.draw(screen)
        self.particles_slider.draw(screen)
        self.stdev_slider.draw(screen)
        self.factor_slider.draw(screen)
        self.motion_noise_xy_slider.draw(screen)
        self.motion_noise_theta_slider.draw(screen)
        
        # Draw buttons
        self.play_button.draw(screen)
        self.reset_sensors_button.draw(screen)
        self.default_sensors_button.draw(screen)
        self.help_button.draw(screen)
        
        # Draw current stats (moved up by 30 pixels)
        stats = [
            "CURRENT SETTINGS:",
            f"Robot: {self.robot.width:.1f} x {self.robot.height:.1f} in",
            f"Sensors: {len(self.robot.sensors)}",
            f"Max Speed: {self.speed_slider.value:.1f} in/s",
            f"Turn Speed: {self.angular_speed_slider.value:.1f} rad/s",
            f"Particles: {self.particles_slider.value:.0f}",
            f"Gaussian StDev: {self.stdev_slider.value:.1f}",
            f"Motion Noise XY: {self.motion_noise_xy_slider.value:.2f}",
            f"Motion Noise Theta: {self.motion_noise_theta_slider.value:.3f}"
        ]
        
        for i, stat in enumerate(stats):
            text = font.render(stat, True, TEXT_COLOR if i == 0 else TEXT_DISABLED)
            screen.blit(text, (self.panels["buttons"].x + 20, self.panels["buttons"].y + 420 + i * 20))  # Was 450, now 420
        
        # Draw help popup if active
        if self.show_help:
            self.draw_help_popup(screen)
    
    def draw_help_popup(self, screen):
        """Draw the scrollable help popup with controls and instructions"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Popup background
        popup_width = 700
        popup_height = 500
        popup_x = (WIDTH - popup_width) // 2
        popup_y = (HEIGHT - popup_height) // 2
        
        pygame.draw.rect(screen, PANEL_BG, (popup_x, popup_y, popup_width, popup_height), border_radius=15)
        pygame.draw.rect(screen, PANEL_BORDER, (popup_x, popup_y, popup_width, popup_height), 3, border_radius=15)
        
        # Title
        title = title_font.render("CONTROLS & INSTRUCTIONS", True, TEXT_COLOR)
        screen.blit(title, (popup_x + (popup_width - title.get_width()) // 2, popup_y + 20))
        
        # Close button
        close_button_rect = pygame.Rect(popup_x + popup_width - 40, popup_y + 10, 30, 30)
        pygame.draw.rect(screen, (200, 50, 50), close_button_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), close_button_rect, 2, border_radius=5)
        close_text = button_font.render("X", True, (255, 255, 255))
        screen.blit(close_text, (close_button_rect.x + 10, close_button_rect.y + 5))
        
        # Create a surface for scrollable content
        content_height = 0
        content_surface = pygame.Surface((popup_width - 40, 800), pygame.SRCALPHA)
        
        content_y = 0
        content_x = 20
        
        sections = [
            ("CUSTOMIZATION:", [
                "• Click on robot to add sensors",
                "• Click on existing sensors to create duplicates",
                "• Drag sensors to reposition",
                "• LEFT/RIGHT arrows to rotate selected sensor",
                "• DELETE or BACKSPACE to remove selected sensor",
                f"• Max sensor range: {SENSOR_MAX_RANGE:.1f} inches",
                "",
                "SENSOR COLORS:",
                "• Yellow: In range (used in MCL)",
                "• Red: Out of range (ignored)",
                "",
                "ROBOT DIMENSIONS:",
                "• Width: Left/Right size (12-18 inches)",
                "• Length: Forward/Backward size (12-18 inches)",
                "• Sensors can be placed anywhere within robot bounds"
            ]),
            ("SIMULATION CONTROLS:", [
                "W/S - Move Forward/Backward",
                "A/D - Turn Left (CCW)/Right (CW)",
                "R - Randomize Robot Position",
                "K - Kidnap (teleport robot without resetting MCL)",
                "SPACE - Reset MCL Particles",
                "H - Toggle Tuning Panel",
                "",
                "COORDINATE SYSTEM:",
                "• 0° points RIGHT",
                "• 90° points UP",
                "• 180° points LEFT",
                "• 270° points DOWN",
                "• Positive angles are counter-clockwise"
            ]),
            ("KIDNAP RECOVERY:", [
                "1. Drive robot to a corner (particles converge)",
                "2. Press K to kidnap (teleport to random position)",
                "3. Drive around in new location",
                "4. MCL will automatically detect the kidnap and:",
                "   • Inject random exploration particles",
                "   • Increase motion noise for better exploration",
                "   • Keep only the best matching particles",
                "   • Converge to new position after some movement",
                "",
                "Note: Kidnap recovery works best when robot moves",
                "and takes sensor readings in the new location."
            ]),
            ("MCL TUNING TIPS:", [
                "• More particles = better accuracy but slower performance",
                "• Lower Gaussian StDev = stricter sensor matching",
                "• Higher Motion Noise = more exploration, less precision",
                "• Adjust parameters in real-time during simulation",
                "",
                "RECOMMENDED SETTINGS:",
                "• Start with 1000-1500 particles",
                "• Gaussian StDev: 1.0-2.0 inches",
                "• Motion Noise XY: 0.1-0.2 inches",
                "• Motion Noise Theta: 0.01-0.02 radians"
            ]),
            ("PARTICLE COLORS:", [
                "RED: Low probability particles",
                "YELLOW/ORANGE: Medium probability particles",
                "GREEN: High probability particles",
                "PURPLE: MCL position estimate (weighted average)",
                "",
                "The colors help visualize particle convergence",
                "Green particles near the robot indicate good localization"
            ]),
            ("TROUBLESHOOTING:", [
                "If particles don't converge:",
                "• Increase motion noise temporarily",
                "• Drive robot to distinctive locations (corners)",
                "• Add more sensors facing different directions",
                "",
                "If kidnap recovery is slow:",
                "• Increase Motion Noise XY/Thet a temporarily",
                "• Reduce Gaussian StDev to be more selective",
                "• Drive robot to get distinctive sensor readings"
            ])
        ]
        
        # Draw all content to the content surface
        for section_title, items in sections:
            # Section title
            section_text = font.render(section_title, True, HIGHLIGHT)
            content_surface.blit(section_text, (content_x, content_y))
            content_y += 25
            
            # Section items
            for item in items:
                item_text = font.render(item, True, TEXT_DISABLED)
                content_surface.blit(item_text, (content_x + 10, content_y))
                content_y += 20
            
            content_y += 15  # Space between sections
        
        # Store content height for scrolling calculations
        self.help_content_height = content_y
        
        # Create a clipping rectangle for the visible area
        clip_rect = pygame.Rect(popup_x + 20, popup_y + 80, popup_width - 60, 400)
        
        # Draw content with clipping
        screen.set_clip(clip_rect)
        screen.blit(content_surface, (popup_x + 20, popup_y + 80 + self.help_scroll_y))
        screen.set_clip(None)
        
        # Draw scrollbar if content is taller than visible area
        if self.help_content_height > 400:
            # Draw scrollbar background
            scrollbar_x = popup_x + popup_width - 30
            scrollbar_rect = pygame.Rect(scrollbar_x, popup_y + 80, 10, 400)
            pygame.draw.rect(screen, SCROLLBAR_COLOR, scrollbar_rect, border_radius=5)
            
            # Calculate scrollbar handle position and size
            visible_height = 400
            content_height = self.help_content_height
            scroll_ratio = -self.help_scroll_y / max(1, content_height - visible_height)
            scrollbar_height = max(30, visible_height * visible_height / max(1, content_height))
            scrollbar_top = popup_y + 80 + scroll_ratio * (visible_height - scrollbar_height)
            
            # Draw scrollbar handle
            handle_rect = pygame.Rect(scrollbar_x, scrollbar_top, 10, scrollbar_height)
            pygame.draw.rect(screen, SCROLLBAR_HANDLE, handle_rect, border_radius=5)
        
        # Draw a close hint at the bottom
        hint_text = font.render("Press ESC or click X to close", True, TEXT_COLOR)
        screen.blit(hint_text, (popup_x + (popup_width - hint_text.get_width()) // 2, popup_y + popup_height - 30))

def world_to_screen(x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates (inches) to screen coordinates"""
    screen_x = FIELD_OFFSET_X + (x + FIELD_WIDTH/2) * SCALE
    screen_y = FIELD_OFFSET_Y + (y + FIELD_HEIGHT/2) * SCALE
    return int(screen_x), int(screen_y)

class FieldScreen:
    """Main field simulation screen"""
    def __init__(self, robot: Robot):
        self.field_map = FieldMap()
        self.robot = robot
        self.randomize_robot_position()
        
        self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta)
        
        # Tuning sliders
        self.particles_slider = Slider(50, HEIGHT - 200, 250, 100, 3000, MCL.n_particles, "Particles", ":d")
        self.stdev_slider = Slider(50, HEIGHT - 170, 250, 0.1, 5.0, MCL.gaussian_stdev, "Gaussian StDev", ":.1f")
        self.factor_slider = Slider(50, HEIGHT - 140, 250, 0.1, 5.0, MCL.gaussian_factor, "Gaussian Factor", ":.1f")
        self.speed_slider = Slider(50, HEIGHT - 110, 250, 20.0, 80.0, robot.max_speed, "Drive Speed", ":.1f")
        self.angular_speed_slider = Slider(50, HEIGHT - 80, 250, 0.5, 10.0, robot.max_angular_speed, "Turn Speed", ":.1f")
        
        # Motion noise sliders
        self.motion_noise_xy_slider = Slider(330, HEIGHT - 200, 250, 0.01, 0.5, 0.15, "Motion Noise XY", ":.2f")
        self.motion_noise_theta_slider = Slider(330, HEIGHT - 170, 250, 0.001, 0.1, 0.015, "Motion Noise Theta", ":.3f")
        
        # Buttons
        self.menu_button = Button(WIDTH - 150, HEIGHT - 50, 120, 40, "MENU", (180, 100, 80))
        self.randomize_button = Button(WIDTH - 150, HEIGHT - 100, 120, 40, "RANDOMIZE", (100, 100, 180))
        self.kidnap_button = Button(WIDTH - 150, HEIGHT - 150, 120, 40, "KIDNAP", KIDNAP_COLOR)
        
        # Control variables
        self.keys = {}
        self.show_tuning = True  # Toggle for showing tuning panel
    
    def randomize_robot_position(self):
        """Place robot at a random position in the field"""
        margin = 20
        self.robot.x = random.uniform(-FIELD_WIDTH/2 + margin, FIELD_WIDTH/2 - margin)
        self.robot.y = random.uniform(-FIELD_HEIGHT/2 + margin, FIELD_HEIGHT/2 - margin)
        self.robot.theta = random.uniform(0, 2 * math.pi)
        
    def kidnap_robot(self):
        """Teleport robot to a random position without resetting MCL"""
        margin = 20
        self.robot.x = random.uniform(-FIELD_WIDTH/2 + margin, FIELD_WIDTH/2 - margin)
        self.robot.y = random.uniform(-FIELD_HEIGHT/2 + margin, FIELD_HEIGHT/2 - margin)
        self.robot.theta = random.uniform(0, 2 * math.pi)
        print(f"Robot kidnapped to: ({self.robot.x:.1f}, {self.robot.y:.1f}, {math.degrees(self.robot.theta):.1f}°)")
        
        # Force MCL to detect kidnap on next update
        self.mcl.kidnapped = True
        self.mcl.kidnap_counter = 5  # High value to ensure detection
    
    def handle_event(self, event):
        if self.show_tuning:
            self.particles_slider.handle_event(event)
            self.stdev_slider.handle_event(event)
            self.factor_slider.handle_event(event)
            self.speed_slider.handle_event(event)
            self.angular_speed_slider.handle_event(event)
            self.motion_noise_xy_slider.handle_event(event)
            self.motion_noise_theta_slider.handle_event(event)
            
            MCL.n_particles = int(self.particles_slider.value)
            MCL.gaussian_stdev = self.stdev_slider.value
            MCL.gaussian_factor = self.factor_slider.value
            self.robot.max_speed = self.speed_slider.value
            self.robot.max_angular_speed = self.angular_speed_slider.value
            
            # Update MCL motion noise from sliders
            self.mcl.motion_noise_xy = self.motion_noise_xy_slider.value
            self.mcl.motion_noise_theta = self.motion_noise_theta_slider.value
        
        if self.menu_button.handle_event(event):
            return "menu"
        
        if self.randomize_button.handle_event(event):
            self.randomize_robot_position()
            self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta)
        
        if self.kidnap_button.handle_event(event):
            self.kidnap_robot()
        
        if event.type == pygame.KEYDOWN:
            self.keys[event.key] = True
            if event.key == pygame.K_r:
                self.randomize_robot_position()
                self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta)
            elif event.key == pygame.K_SPACE:
                self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta)
            elif event.key == pygame.K_k:
                self.kidnap_robot()
            elif event.key == pygame.K_h:  # Toggle tuning panel
                self.show_tuning = not self.show_tuning
        
        elif event.type == pygame.KEYUP:
            self.keys[event.key] = False
        
        return None
    
    def update(self, dt: float):
        # Handle robot controls with corrected turning
        self.robot.speed = 0
        self.robot.angular_speed = 0
        
        if self.keys.get(pygame.K_w):
            self.robot.speed = self.robot.max_speed  # Forward
        if self.keys.get(pygame.K_s):
            self.robot.speed = -self.robot.max_speed  # Backward
        if self.keys.get(pygame.K_a):  # A = turn left (counter-clockwise)
            self.robot.angular_speed = -self.robot.max_angular_speed
        if self.keys.get(pygame.K_d):  # D = turn right (clockwise)
            self.robot.angular_speed = self.robot.max_angular_speed
        
        # Update robot position
        self.robot.update(dt)
        
        # Keep robot within bounds
        field_limit = FIELD_WIDTH/2 - WALL_THICKNESS - self.robot.width/2 - 2
        self.robot.x = max(-field_limit, min(field_limit, self.robot.x))
        self.robot.y = max(-field_limit, min(field_limit, self.robot.y))
        
        # Update robot sensors
        self.robot.update_sensors(self.field_map)
        
        # Get deltas from robot's absolute position change
        dx, dy, dtheta = self.robot.get_deltas()
        
        # Run MCL with absolute position deltas
        self.mcl.run(self.robot.sensors, dx, dy, dtheta)
    
    def draw(self, screen):
        screen.fill(BACKGROUND)
        
        field_rect = pygame.Rect(
            FIELD_OFFSET_X, FIELD_OFFSET_Y,
            FIELD_WIDTH * SCALE, FIELD_HEIGHT * SCALE
        )
        
        if self.field_map.background_image:
            screen.blit(self.field_map.background_image, field_rect)
        else:
            pygame.draw.rect(screen, FIELD_BG, field_rect)
        
        pygame.draw.rect(screen, WALL_COLOR, field_rect, 3)
        
        for obstacle in self.field_map.obstacles:
            x1, y1 = world_to_screen(obstacle.x_min, obstacle.y_min)
            x2, y2 = world_to_screen(obstacle.x_max, obstacle.y_max)
            obstacle_rect = pygame.Rect(x1, y1, x2-x1, y2-y1)
            pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle_rect)
        
        # Draw particles FIRST
        self.draw_particles(screen)
        
        # Draw robot
        self.draw_robot(screen)
        
        # Draw estimated pose
        self.draw_estimated_pose(screen)
        
        # Draw UI
        self.draw_ui(screen)
    
    def draw_particles(self, screen):
        """Draw particles colored by weight"""
        if not self.mcl.particles:
            return
        
        alive_particles = [p for p in self.mcl.particles if p.alive]
        if not alive_particles:
            return
            
        weights = [p.weight for p in alive_particles]
        max_weight = max(weights) if weights else 1.0
        min_weight = min(weights) if weights else 0.0
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        
        for particle in alive_particles:
            if weight_range > 0:
                normalized_weight = (particle.weight - min_weight) / weight_range
            else:
                normalized_weight = 0.5
            
            if normalized_weight < 0.5:
                r = PARTICLE_LOW[0] + (PARTICLE_MID[0] - PARTICLE_LOW[0]) * (normalized_weight * 2)
                g = PARTICLE_LOW[1] + (PARTICLE_MID[1] - PARTICLE_LOW[1]) * (normalized_weight * 2)
                b = PARTICLE_LOW[2] + (PARTICLE_MID[2] - PARTICLE_LOW[2]) * (normalized_weight * 2)
            else:
                r = PARTICLE_MID[0] + (PARTICLE_HIGH[0] - PARTICLE_MID[0]) * ((normalized_weight - 0.5) * 2)
                g = PARTICLE_MID[1] + (PARTICLE_HIGH[1] - PARTICLE_MID[1]) * ((normalized_weight - 0.5) * 2)
                b = PARTICLE_MID[2] + (PARTICLE_HIGH[2] - PARTICLE_MID[2]) * ((normalized_weight - 0.5) * 2)
            
            color = (int(r), int(g), int(b))
            
            x, y = world_to_screen(particle.x, particle.y)
            pygame.draw.circle(screen, color, (x, y), 2)
    
    def draw_robot(self, screen):
        """Draw the robot and its sensors"""
        corners = self.robot.get_corners()
        screen_corners = [world_to_screen(x, y) for x, y in corners]
        
        robot_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(robot_surface, ROBOT_COLOR, screen_corners)
        pygame.draw.polygon(robot_surface, ROBOT_OUTLINE, screen_corners, 2)
        
        center_x, center_y = world_to_screen(self.robot.x, self.robot.y)
        pygame.draw.circle(robot_surface, (255, 255, 255, 200), (center_x, center_y), 4)
        
        screen.blit(robot_surface, (0, 0))
        
        # Draw heading (0° is right)
        heading_length = 25
        heading_x = center_x + heading_length * math.cos(self.robot.theta)
        heading_y = center_y + heading_length * math.sin(self.robot.theta)
        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (heading_x, heading_y), 3)
        
        # Draw sensors and rays
        for sensor in self.robot.sensors:
            sensor_pos = sensor.get_global_position(self.robot.x, self.robot.y, self.robot.theta)
            sensor_screen = world_to_screen(sensor_pos[0], sensor_pos[1])
            
            sensor_color = SENSOR_OUT_RANGE if not sensor.in_range else SENSOR_COLOR
            pygame.draw.circle(screen, sensor_color, sensor_screen, 6)
            
            sensor_angle = sensor.get_global_angle(self.robot.theta)
            end_x = sensor_pos[0] + sensor.distance * math.cos(sensor_angle)
            end_y = sensor_pos[1] + sensor.distance * math.sin(sensor_angle)
            end_screen = world_to_screen(end_x, end_y)
            
            ray_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            ray_color = RAY_OUT_RANGE if not sensor.in_range else RAY_COLOR
            pygame.draw.line(ray_surface, ray_color, sensor_screen, end_screen, 2)
            screen.blit(ray_surface, (0, 0))
            
            if sensor.in_range and sensor.distance < SENSOR_MAX_RANGE:
                pygame.draw.circle(screen, (255, 50, 50), end_screen, 4)
    
    def draw_estimated_pose(self, screen):
        """Draw the estimated pose from MCL"""
        if self.mcl.average_pose:
            x, y, theta = self.mcl.average_pose
            screen_x, screen_y = world_to_screen(x, y)
            
            pygame.draw.circle(screen, ESTIMATE_COLOR, (screen_x, screen_y), 10)
            pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 10, 2)
            
            # Draw heading (0° is right)
            heading_length = 20
            heading_x = screen_x + heading_length * math.cos(theta)
            heading_y = screen_y + heading_length * math.sin(theta)
            pygame.draw.line(screen, ESTIMATE_COLOR, (screen_x, screen_y), (heading_x, heading_y), 4)
    
    def draw_ui(self, screen):
        """Draw UI elements"""
        title = title_font.render("FIELD SIMULATION", True, TEXT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        
        # Robot info panel (70 pixels taller - from 280 to 350)
        info_panel = pygame.Rect(20, 20, 320, 350)
        pygame.draw.rect(screen, PANEL_BG, info_panel, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, info_panel, 3, border_radius=10)
        
        info_y = 40
        info_lines = [
            "ROBOT INFO:",
            f"Position: ({self.robot.x:.1f}, {self.robot.y:.1f})",
            f"Heading: {math.degrees(self.robot.theta):.1f}°",
            f"Size: {self.robot.width:.1f} x {self.robot.height:.1f}",
            f"Sensors: {len(self.robot.sensors)}",
        ]
        
        for i, line in enumerate(info_lines):
            color = TEXT_COLOR if i == 0 else TEXT_DISABLED
            text = font.render(line, True, color)
            screen.blit(text, (40, info_y))
            info_y += 20
        
        # Sensor readings (with more space)
        info_y += 10
        sensor_title = font.render("SENSOR READINGS:", True, TEXT_COLOR)
        screen.blit(sensor_title, (40, info_y))
        info_y += 20
        
        sensors_in_range = 0
        for i, sensor in enumerate(self.robot.sensors):
            if sensor.in_range:
                sensors_in_range += 1
            status = "IN" if sensor.in_range else "OUT"
            color = TEXT_COLOR if sensor.in_range else SENSOR_OUT_RANGE
            reading = f"{sensor.distance:.1f}" if sensor.in_range else ">78.7"
            sensor_text = font.render(f"S{i+1}: {reading}in ({status})", True, color)
            screen.blit(sensor_text, (50, info_y))
            info_y += 18
        
        # MCL info
        info_y += 15
        mcl_lines = [
            "MCL INFO:",
            f"Particles: {len(self.mcl.particles)}",
            f"Used sensors: {sensors_in_range}/{len(self.robot.sensors)}",
            f"Estimate Error: {math.sqrt((self.robot.x - self.mcl.average_pose[0])**2 + (self.robot.y - self.mcl.average_pose[1])**2):.2f} in",
            f"Kidnap Counter: {self.mcl.kidnap_counter}"
        ]
        
        for i, line in enumerate(mcl_lines):
            color = TEXT_COLOR if i == 0 else TEXT_DISABLED
            text = font.render(line, True, color)
            screen.blit(text, (40, info_y))
            info_y += 20
        
        # Kidnap status
        if self.mcl.kidnapped:
            kidnap_text = font.render("KIDNAP RECOVERY ACTIVE!", True, KIDNAP_ACTIVE_COLOR)
            screen.blit(kidnap_text, (40, info_y + 5))
        
        # Particle color legend (moved down to accommodate taller robot info box)
        legend_panel = pygame.Rect(20, 380, 320, 120)  # From 310 to 380 (down 70 pixels)
        pygame.draw.rect(screen, PANEL_BG, legend_panel, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, legend_panel, 3, border_radius=10)
        
        legend_title = font.render("PARTICLE COLORS:", True, TEXT_COLOR)
        screen.blit(legend_title, (40, 395))
        
        # Draw color swatches and labels
        color_items = [
            (PARTICLE_LOW, "Low probability"),
            (PARTICLE_MID, "Medium probability"),
            (PARTICLE_HIGH, "High probability"),
            (ESTIMATE_COLOR, "MCL estimate")
        ]
        
        for i, (color, label) in enumerate(color_items):
            pygame.draw.rect(screen, color, (50, 420 + i * 20, 12, 12))
            text = font.render(label, True, TEXT_DISABLED)
            screen.blit(text, (70, 420 + i * 20))
        
        # Tuning panel (taller to prevent text overlap)
        if self.show_tuning:
            tuning_panel = pygame.Rect(20, HEIGHT - 220, 580, 200)  # Taller panel
            pygame.draw.rect(screen, PANEL_BG, tuning_panel, border_radius=10)
            pygame.draw.rect(screen, PANEL_BORDER, tuning_panel, 3, border_radius=10)
            
            tuning_title = font.render("REAL-TIME TUNING (Press H to hide)", True, TEXT_COLOR)
            screen.blit(tuning_title, (40, HEIGHT - 215))
            
            self.particles_slider.draw(screen)
            self.stdev_slider.draw(screen)
            self.factor_slider.draw(screen)
            self.speed_slider.draw(screen)
            self.angular_speed_slider.draw(screen)
            self.motion_noise_xy_slider.draw(screen)
            self.motion_noise_theta_slider.draw(screen)
        else:
            # Show a small indicator that tuning is hidden
            tuning_title = font.render("Tuning hidden (Press H to show)", True, TEXT_DISABLED)
            screen.blit(tuning_title, (40, HEIGHT - 30))
        
        # Buttons
        self.menu_button.draw(screen)
        self.randomize_button.draw(screen)
        self.kidnap_button.draw(screen)

def main():
    current_screen = "customization"
    customization_screen = RobotCustomizationScreen()
    field_screen = None
    
    running = True
    last_time = pygame.time.get_ticks()
    
    while running:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0
        last_time = current_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if current_screen == "customization":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    customization_screen.show_help = False
                    customization_screen.help_scroll_y = 0
                
                result = customization_screen.handle_event(event)
                if result == "play":
                    customization_screen.update()
                    field_screen = FieldScreen(customization_screen.robot)
                    current_screen = "field"
            
            elif current_screen == "field":
                result = field_screen.handle_event(event)
                if result == "menu":
                    current_screen = "customization"
        
        if current_screen == "customization":
            customization_screen.update()
            customization_screen.draw(screen)
        
        elif current_screen == "field" and field_screen:
            field_screen.update(dt)
            field_screen.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
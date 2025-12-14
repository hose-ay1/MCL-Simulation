import pygame
import math
import random
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MCL Robot Localization Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)
button_font = pygame.font.SysFont('Arial', 18)

BACKGROUND = (25, 25, 35)
CUSTOM_BG = (40, 40, 50)
PANEL_BG = (50, 50, 65)
PANEL_BORDER = (80, 80, 100)
FIELD_BG = (240, 240, 235)
WALL_COLOR = (60, 60, 80)
OBSTACLE_COLOR = (200, 100, 100)
ROBOT_COLOR = (50, 150, 220, 100)
ROBOT_OUTLINE = (50, 150, 220, 200)
SENSOR_COLOR = (240, 200, 60)
SENSOR_SELECTED_COLOR = (255, 220, 40)
SENSOR_HALO = (255, 220, 40, 100)
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
HELP_COLOR = (100, 100, 220)
SCROLLBAR_COLOR = (80, 80, 100)
SCROLLBAR_HANDLE = (120, 120, 140)

FIELD_WIDTH = 144
FIELD_HEIGHT = 144
SCALE = 4.5
FIELD_OFFSET_X = 350
FIELD_OFFSET_Y = 100

SENSOR_MAX_RANGE = 78.74016
SENSOR_MIN_RANGE = 0.0
WALL_THICKNESS = 2.5
OBSTACLE_THICKNESS = 0.125

@dataclass
class Rectangle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class FieldMap:
    def __init__(self):
        self.width = FIELD_WIDTH
        self.height = FIELD_HEIGHT
        
        self.walls = [
            Rectangle(-self.width/2, -self.height/2, self.width/2, -self.height/2 + WALL_THICKNESS),
            Rectangle(-self.width/2, self.height/2 - WALL_THICKNESS, self.width/2, self.height/2),
            Rectangle(-self.width/2, -self.height/2, -self.width/2 + WALL_THICKNESS, self.height/2),
            Rectangle(self.width/2 - WALL_THICKNESS, -self.height/2, self.width/2, self.height/2),
        ]
        self.obstacles = []
        self.lines = []
        self.create_triangular_objects()
        self.background_image = None
        self.load_background()
    
    def load_background(self):
        try:
            image_paths = ["field.png", "./field.png", os.path.join(os.path.dirname(__file__), "field.png")]
            for path in image_paths:
                if os.path.exists(path):
                    self.background_image = pygame.image.load(path)
                    target_width = int(FIELD_WIDTH * SCALE)
                    target_height = int(FIELD_HEIGHT * SCALE)
                    self.background_image = pygame.transform.scale(self.background_image, (target_width, target_height))
                    break
        except:
            self.background_image = None
    
    def add_line_as_obstacle(self, x1, y1, x2, y2, thickness=OBSTACLE_THICKNESS):
        self.lines.append(((x1, y1), (x2, y2)))
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        dx /= length
        dy /= length
        px = dy
        py = -dx
        half = thickness / 2.0
        corners = [
            (x1 + px * half, y1 + py * half),
            (x1 - px * half, y1 - py * half),
            (x2 + px * half, y2 + py * half),
            (x2 - px * half, y2 - py * half),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        self.obstacles.append(Rectangle(min(xs), min(ys), max(xs), max(ys)))
    
    def create_triangular_objects(self):
        lines_q1 = [(23.5, 49.5, 21, 47), (21, 47, 23.5, 44.5)]
        quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for x_sign, y_sign in quadrants:
            for x1, y1, x2, y2 in lines_q1:
                self.add_line_as_obstacle(x1 * x_sign, y1 * y_sign, x2 * x_sign, y2 * y_sign)
    
    def raycast(self, origin_x, origin_y, angle, max_range=SENSOR_MAX_RANGE):
        dx = math.sin(angle)
        dy = math.cos(angle)
        eps = 0.001
        min_distance = max_range
        infinity = float('inf')
        
        for rect in self.walls + self.obstacles:
            t_x1 = (rect.x_min - origin_x) / dx if abs(dx) > eps else infinity
            t_x2 = (rect.x_max - origin_x) / dx if abs(dx) > eps else infinity
            t_y1 = (rect.y_min - origin_y) / dy if abs(dy) > eps else infinity
            t_y2 = (rect.y_max - origin_y) / dy if abs(dy) > eps else infinity
            
            tmin = max(min(t_x1, t_x2), min(t_y1, t_y2))
            tmax = min(max(t_x1, t_x2), max(t_y1, t_y2))
            
            if tmax >= 0 and tmin <= tmax:
                distance = max(tmin, 0)
                if 0 < distance < min_distance:
                    min_distance = distance
        
        in_range = min_distance < max_range - 0.1
        return min_distance, in_range

class Sensor:
    def __init__(self, offset_x, offset_y, angle_offset):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.angle_offset_rad = angle_offset
        self.distance = 0.0
        self.max_range = SENSOR_MAX_RANGE
        self.in_range = True
        self.cut_out = False
    
    def get_global_position(self, robot_x, robot_y, robot_theta):
        cos_theta = math.cos(robot_theta)
        sin_theta = math.sin(robot_theta)
        sensor_x = robot_x + self.offset_x * cos_theta + self.offset_y * sin_theta
        sensor_y = robot_y - self.offset_x * sin_theta + self.offset_y * cos_theta
        return sensor_x, sensor_y
    
    def get_global_angle(self, robot_theta):
        sensor_angle = robot_theta + self.angle_offset_rad
        sensor_angle %= (2 * math.pi)
        return sensor_angle
    
    def update(self, robot_x, robot_y, robot_theta, field_map):
        sensor_x, sensor_y = self.get_global_position(robot_x, robot_y, robot_theta)
        sensor_angle = self.get_global_angle(robot_theta)
        
        measured_distance, self.in_range = field_map.raycast(sensor_x, sensor_y, sensor_angle, self.max_range)
        self.cut_out = not self.in_range
        
        if self.in_range:
            self.distance = measured_distance + random.gauss(0, 0.5)
            self.distance = max(SENSOR_MIN_RANGE, min(self.distance, self.max_range))
        else:
            self.distance = self.max_range

class Robot:
    def __init__(self, width=12.5, length=15.0):
        self.width = max(12, min(18, width))
        self.length = max(12, min(18, length))
        self.x = 0
        self.y = 0
        self.theta = 0
        self.speed = 0.0
        self.angular_speed = 0.0
        self.max_speed = 40.0
        self.max_angular_speed = 3.0
        self.sensors = []
        self.sensor_positions = []
        self.prev_x = 0
        self.prev_y = 0
        self.prev_theta = 0
    
    def set_size(self, width, length):
        self.width = max(12, min(18, width))
        self.length = max(12, min(18, length))
    
    def add_sensor(self, offset_x, offset_y, angle_offset):
        max_offset_x = self.width / 2 + 2
        max_offset_y = self.length / 2 + 2
        offset_x = max(-max_offset_x, min(max_offset_x, offset_x))
        offset_y = max(-max_offset_y, min(max_offset_y, offset_y))
        self.sensor_positions.append((offset_x, offset_y, angle_offset))
        self.sensors.append(Sensor(offset_x, offset_y, angle_offset))
    
    def clear_sensors(self):
        self.sensors = []
        self.sensor_positions = []
    
    def update(self, dt):
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
        self.x += self.speed * math.sin(self.theta) * dt
        self.y += self.speed * math.cos(self.theta) * dt
        self.theta += self.angular_speed * dt
        self.theta %= (2 * math.pi)
    
    def get_deltas(self):
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        dtheta = self.theta - self.prev_theta
        while dtheta > math.pi:
            dtheta -= 2 * math.pi
        while dtheta < -math.pi:
            dtheta += 2 * math.pi
        return dx, dy, dtheta
    
    def update_sensors(self, field_map):
        for sensor in self.sensors:
            sensor.update(self.x, self.y, self.theta, field_map)
    
    def get_corners(self):
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        half_w = self.width / 2
        half_l = self.length / 2
        corners_local = [
            (-half_w, half_l),
            (half_w, half_l),
            (half_w, -half_l),
            (-half_w, -half_l)
        ]
        corners_global = []
        for x_local, y_local in corners_local:
            x_global = self.x + x_local * cos_theta + y_local * sin_theta
            y_global = self.y - x_local * sin_theta + y_local * cos_theta
            corners_global.append((x_global, y_global))
        return corners_global

class Particle:
    def __init__(self, x=0, y=0, theta=0, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta % (2 * math.pi)
        self.weight = weight
        self.alive = True
    
    def is_in_field(self):
        return (-72 < self.x < 72) and (-72 < self.y < 72)
    
    def predict(self, delta_x, delta_y, delta_theta, motion_noise_xy, motion_noise_theta, random_generator):
        theta_noise = random.gauss(0, motion_noise_theta)
        self.theta += delta_theta + theta_noise
        self.theta %= (2 * math.pi)
        
        x_noise = random.gauss(0, motion_noise_xy)
        y_noise = random.gauss(0, motion_noise_xy)
        self.x += delta_x + x_noise
        self.y += delta_y + y_noise
        self.alive = self.is_in_field()
    
    def measurement_probability(self, sensors, field_map, gaussian_stddev, heading_offset_rad):
        if not sensors or not self.alive:
            return 0.0
        
        log_sum = 0.0
        valid_sensor_count = 0
        
        for sensor in sensors:
            sensor_x, sensor_y = sensor.get_global_position(self.x, self.y, self.theta + heading_offset_rad)
            sensor_angle = sensor.get_global_angle(self.theta + heading_offset_rad)
            
            expected_dist, expected_in_range = field_map.raycast(sensor_x, sensor_y, sensor_angle, sensor.max_range)
            
            if sensor.cut_out and not expected_in_range:
                prob = 0.1
            elif not sensor.cut_out and expected_in_range:
                error = abs(sensor.distance - expected_dist)
                error = min(error, 3.0 * gaussian_stddev)
                prob = math.exp(-0.5 * (error * error) / (gaussian_stddev * gaussian_stddev))
            else:
                prob = 0.001
            
            prob = max(prob, 1e-10)
            log_sum += math.log(prob)
            valid_sensor_count += 1
        
        if valid_sensor_count == 0:
            return 0.001
        
        return math.exp(log_sum / valid_sensor_count)
    
    def clone(self):
        return Particle(self.x, self.y, self.theta, self.weight)

class MCL:
    def __init__(self, field_map, robot_x=0, robot_y=0, robot_theta=0, num_particles=1000, 
                 gaussian_stddev=1.5, gaussian_factor=1.0, motion_noise_xy=0.15, motion_noise_theta=0.015):
        self.field_map = field_map
        self.particles = []
        self.average_pose = (0.0, 0.0, 0.0)
        self.num_particles = num_particles
        self.gaussian_stddev = gaussian_stddev
        self.gaussian_factor = gaussian_factor
        self.motion_noise_xy = motion_noise_xy
        self.motion_noise_theta = motion_noise_theta
        self.initialize_around_pose(robot_x, robot_y, robot_theta, 5.0)
    
    def initialize_uniform(self):
        self.particles = []
        for _ in range(self.num_particles):
            x = random.uniform(-self.field_map.width/2 + 5, self.field_map.width/2 - 5)
            y = random.uniform(-self.field_map.height/2 + 5, self.field_map.height/2 - 5)
            theta = random.uniform(0, 2 * math.pi)
            self.particles.append(Particle(x, y, theta))
    
    def initialize_around_pose(self, x, y, theta, variance=5.0):
        self.particles = []
        created_particles = 0
        while created_particles < self.num_particles:
            px = x + random.gauss(0, variance)
            py = y + random.gauss(0, variance)
            ptheta = theta + random.gauss(0, 0.1)
            particle = Particle(px, py, ptheta)
            if particle.is_in_field():
                self.particles.append(particle)
                created_particles += 1
    
    def update_step(self, delta_x, delta_y, delta_theta):
        for particle in self.particles:
            particle.predict(delta_x, delta_y, delta_theta, self.motion_noise_xy, self.motion_noise_theta, random)
        
        self.particles = [p for p in self.particles if p.alive]
        
        while len(self.particles) < self.num_particles:
            if self.particles:
                random_particle = random.choice(self.particles)
                self.particles.append(random_particle.clone())
            else:
                x = random.uniform(-self.field_map.width/2 + 5, self.field_map.width/2 - 5)
                y = random.uniform(-self.field_map.height/2 + 5, self.field_map.height/2 - 5)
                theta = random.uniform(0, 2 * math.pi)
                self.particles.append(Particle(x, y, theta))
    
    def resample_step(self, sensors, heading_offset_rad=0.0):
        total_weight = 0.0
        for particle in self.particles:
            particle.weight = particle.measurement_probability(sensors, self.field_map, self.gaussian_stddev, heading_offset_rad) * self.gaussian_factor
            total_weight += particle.weight
        
        if total_weight <= 0:
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)
            total_weight = 1.0
        
        for particle in self.particles:
            particle.weight /= total_weight
        
        eff_n = 1.0 / sum(p.weight ** 2 for p in self.particles)
        
        if eff_n < self.num_particles * 0.5:
            new_particles = []
            r = random.uniform(0, 1 / self.num_particles)
            c = self.particles[0].weight
            i = 0
            
            for m in range(self.num_particles):
                u = r + m / self.num_particles
                while u > c and i < len(self.particles) - 1:
                    i += 1
                    c += self.particles[i].weight
                if i < len(self.particles):
                    new_particles.append(self.particles[i].clone())
            
            self.particles = new_particles
        
        self.update_average_pose()
    
    def update_average_pose(self):
        alive_particles = [p for p in self.particles if p.alive]
        if not alive_particles:
            return
        
        total_weight = sum(p.weight for p in alive_particles)
        if total_weight == 0:
            return
        
        x = sum(p.x * p.weight for p in alive_particles) / total_weight
        y = sum(p.y * p.weight for p in alive_particles) / total_weight
        sin_sum = sum(math.sin(p.theta) * p.weight for p in alive_particles)
        cos_sum = sum(math.cos(p.theta) * p.weight for p in alive_particles)
        theta = math.atan2(sin_sum, cos_sum)
        
        self.average_pose = (x, y, theta)
    
    def run(self, sensors, delta_x, delta_y, delta_theta, heading_offset_rad=0.0):
        self.update_step(delta_x, delta_y, delta_theta)
        self.resample_step(sensors, heading_offset_rad)
        return self.average_pose

class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val, label, value_format=":.2f"):
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
    def __init__(self, x, y, width, height, label, color=BUTTON_COLOR, text_color=BUTTON_TEXT):
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
    def __init__(self):
        self.robot = Robot()
        self.dragging_sensor = None
        self.selected_sensor = None
        self.show_help = False
        self.help_scroll_y = 0
        self.scroll_dragging = False
        self.scroll_start_y = 0
        self.help_content_height = 0
        
        self.panels = {
            "robot": pygame.Rect(50, 100, 350, 620),
            "params": pygame.Rect(420, 100, 350, 620),
            "buttons": pygame.Rect(790, 100, 350, 620)
        }
        
        self.width_slider = Slider(70, 160, 250, 12, 18, 12.5, "Robot Width", ":.1f")
        self.length_slider = Slider(70, 210, 250, 12, 18, 15.0, "Robot Length", ":.1f")
        self.speed_slider = Slider(70, 260, 250, 20.0, 60.0, 40.0, "Max Speed", ":.1f")
        self.angular_speed_slider = Slider(70, 310, 250, 0.5, 10.0, 3.0, "Turn Speed", ":.1f")
        self.particles_slider = Slider(440, 160, 250, 100, 3000, 1000, "Number of Particles", ":d")
        self.stdev_slider = Slider(440, 210, 250, 0.1, 5.0, 1.5, "Gaussian StDev", ":.1f")
        self.factor_slider = Slider(440, 260, 250, 0.1, 5.0, 1.0, "Gaussian Factor", ":.1f")
        self.motion_noise_xy_slider = Slider(440, 310, 250, 0.01, 0.5, 0.15, "Motion Noise XY", ":.2f")
        self.motion_noise_theta_slider = Slider(440, 360, 250, 0.001, 0.1, 0.015, "Motion Noise Theta", ":.3f")
        
        self.play_button = Button(810, 200, 300, 60, "PLAY SIMULATION", (80, 180, 90))
        self.reset_sensors_button = Button(810, 280, 300, 50, "CLEAR ALL SENSORS", (180, 100, 80))
        self.default_sensors_button = Button(810, 350, 300, 50, "SET DEFAULT SENSORS", (100, 100, 180))
        self.help_button = Button(810, 420, 300, 50, "CONTROLS & INSTRUCTIONS", HELP_COLOR)
        
        self.set_default_sensors()
    
    def handle_event(self, event):
        if self.show_help:
            popup_width = 700
            popup_height = 500
            popup_x = (WIDTH - popup_width) // 2
            popup_y = (HEIGHT - popup_height) // 2
            
            mouse_x, mouse_y = pygame.mouse.get_pos()
            popup_clicked = (popup_x <= mouse_x <= popup_x + popup_width and 
                            popup_y <= mouse_y <= popup_y + popup_height)
            
            if event.type == pygame.MOUSEWHEEL:
                if popup_clicked:
                    self.help_scroll_y -= event.y * 20
                    self.help_scroll_y = max(-self.help_content_height + 400, min(0, self.help_scroll_y))
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                scrollbar_x = popup_x + popup_width - 20
                scrollbar_width = 10
                content_height = self.help_content_height
                visible_height = 400
                scroll_ratio = -self.help_scroll_y / max(1, content_height - visible_height)
                scrollbar_height = max(30, visible_height * visible_height / max(1, content_height))
                scrollbar_top = popup_y + 80 + scroll_ratio * (visible_height - scrollbar_height)
                
                if (scrollbar_x <= mouse_x <= scrollbar_x + scrollbar_width and 
                    scrollbar_top <= mouse_y <= scrollbar_top + scrollbar_height):
                    self.scroll_dragging = True
                    self.scroll_start_y = mouse_y
                
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
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.show_help = False
                self.help_scroll_y = 0
            return None
        
        self.width_slider.handle_event(event)
        self.length_slider.handle_event(event)
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
            self.dragging_sensor = None
        
        if self.default_sensors_button.handle_event(event):
            self.set_default_sensors()
            self.selected_sensor = 0
            self.dragging_sensor = None
        
        if self.help_button.handle_event(event):
            self.show_help = True
            self.help_scroll_y = 0
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self.panels["robot"].collidepoint(mouse_x, mouse_y):
                center_x = self.panels["robot"].centerx
                center_y = 450
                sensor_clicked = False
                for i, sensor in enumerate(self.robot.sensors):
                    sensor_x = center_x + sensor.offset_x * 8
                    sensor_y = center_y - sensor.offset_y * 8
                    if math.sqrt((mouse_x - sensor_x)**2 + (mouse_y - sensor_y)**2) < 15:
                        self.selected_sensor = i
                        self.dragging_sensor = i
                        sensor_clicked = True
                        break
                if not sensor_clicked and (center_y - 100 < mouse_y < center_y + 100):
                    offset_x = (mouse_x - center_x) / 8
                    offset_y = (center_y - mouse_y) / 8
                    max_offset_x = self.robot.width / 2
                    max_offset_y = self.robot.length / 2
                    if abs(offset_x) <= max_offset_x and abs(offset_y) <= max_offset_y:
                        self.robot.add_sensor(offset_x, offset_y, 0)
                        self.selected_sensor = len(self.robot.sensors) - 1
                        self.dragging_sensor = self.selected_sensor
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_sensor = None
        
        elif event.type == pygame.MOUSEMOTION and self.dragging_sensor is not None:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            center_x = self.panels["robot"].centerx
            center_y = 450
            offset_x = (mouse_x - center_x) / 8
            offset_y = (center_y - mouse_y) / 8
            max_offset_x = self.robot.width / 2
            max_offset_y = self.robot.length / 2
            offset_x = max(-max_offset_x, min(max_offset_x, offset_x))
            offset_y = max(-max_offset_y, min(max_offset_y, offset_y))
            self.robot.sensors[self.dragging_sensor].offset_x = offset_x
            self.robot.sensors[self.dragging_sensor].offset_y = offset_y
            self.robot.sensor_positions[self.dragging_sensor] = (offset_x, offset_y, 
                self.robot.sensors[self.dragging_sensor].angle_offset_rad)
        
        elif event.type == pygame.KEYDOWN:
            if self.selected_sensor is not None and self.selected_sensor < len(self.robot.sensors):
                if event.key == pygame.K_LEFT:
                    self.robot.sensors[self.selected_sensor].angle_offset_rad -= math.pi/8
                    self.robot.sensor_positions[self.selected_sensor] = (
                        self.robot.sensors[self.selected_sensor].offset_x,
                        self.robot.sensors[self.selected_sensor].offset_y,
                        self.robot.sensors[self.selected_sensor].angle_offset_rad
                    )
                elif event.key == pygame.K_RIGHT:
                    self.robot.sensors[self.selected_sensor].angle_offset_rad += math.pi/8
                    self.robot.sensor_positions[self.selected_sensor] = (
                        self.robot.sensors[self.selected_sensor].offset_x,
                        self.robot.sensors[self.selected_sensor].offset_y,
                        self.robot.sensors[self.selected_sensor].angle_offset_rad
                    )
                elif event.key in [pygame.K_DELETE, pygame.K_BACKSPACE]:
                    self.robot.sensors.pop(self.selected_sensor)
                    self.robot.sensor_positions.pop(self.selected_sensor)
                    if self.robot.sensors:
                        self.selected_sensor = min(self.selected_sensor, len(self.robot.sensors) - 1)
                    else:
                        self.selected_sensor = None
        
        return None
    
    def set_default_sensors(self):
        self.robot.clear_sensors()
        self.robot.add_sensor(0, self.robot.length/2, 0)
        self.robot.add_sensor(-self.robot.width/2, 0, -math.pi/2)
        self.robot.add_sensor(0, -self.robot.length/2, math.pi)
        self.robot.add_sensor(self.robot.width/2, 0, math.pi/2)
        self.selected_sensor = 0
    
    def update(self):
        self.robot.set_size(self.width_slider.value, self.length_slider.value)
        self.robot.max_speed = self.speed_slider.value
        self.robot.max_angular_speed = self.angular_speed_slider.value
        
        self.num_particles = int(self.particles_slider.value)
        self.gaussian_stddev = self.stdev_slider.value
        self.gaussian_factor = self.factor_slider.value
        self.motion_noise_xy = self.motion_noise_xy_slider.value
        self.motion_noise_theta = self.motion_noise_theta_slider.value
    
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
        
        center_x, center_y = self.panels["robot"].centerx, 450
        scale = 8
        robot_surface = pygame.Surface((self.robot.width * scale, self.robot.length * scale), pygame.SRCALPHA)
        pygame.draw.rect(robot_surface, ROBOT_COLOR, robot_surface.get_rect(), border_radius=5)
        pygame.draw.rect(robot_surface, ROBOT_OUTLINE, robot_surface.get_rect(), 2, border_radius=5)
        screen.blit(robot_surface, (center_x - self.robot.width/2 * scale, center_y - self.robot.length/2 * scale))
        
        heading_length = 15
        heading_x = center_x
        heading_y = center_y - heading_length
        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (heading_x, heading_y), 3)
        pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 4)
        
        for i, sensor in enumerate(self.robot.sensors):
            sensor_x = center_x + sensor.offset_x * scale
            sensor_y = center_y - sensor.offset_y * scale
            if i == self.selected_sensor:
                halo_radius = 15
                halo_surface = pygame.Surface((halo_radius*2, halo_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(halo_surface, SENSOR_HALO, (halo_radius, halo_radius), halo_radius)
                screen.blit(halo_surface, (sensor_x - halo_radius, sensor_y - halo_radius))
            
            color = SENSOR_SELECTED_COLOR if i == self.selected_sensor else SENSOR_COLOR
            radius = 8 if i == self.selected_sensor else 6
            pygame.draw.circle(screen, color, (int(sensor_x), int(sensor_y)), radius)
            
            dir_length = 12
            dir_x = sensor_x + dir_length * math.sin(sensor.angle_offset_rad)
            dir_y = sensor_y - dir_length * math.cos(sensor.angle_offset_rad)
            pygame.draw.line(screen, color, (sensor_x, sensor_y), (dir_x, dir_y), 2)
            
            num_text = font.render(str(i+1), True, (0, 0, 0))
            screen.blit(num_text, (sensor_x - 4, sensor_y - 6))
        
        self.width_slider.draw(screen)
        self.length_slider.draw(screen)
        self.speed_slider.draw(screen)
        self.angular_speed_slider.draw(screen)
        self.particles_slider.draw(screen)
        self.stdev_slider.draw(screen)
        self.factor_slider.draw(screen)
        self.motion_noise_xy_slider.draw(screen)
        self.motion_noise_theta_slider.draw(screen)
        self.play_button.draw(screen)
        self.reset_sensors_button.draw(screen)
        self.default_sensors_button.draw(screen)
        self.help_button.draw(screen)
        
        stats = [
            "CURRENT SETTINGS:",
            f"Robot: {self.robot.width:.1f} x {self.robot.length:.1f} in",
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
            screen.blit(text, (self.panels["buttons"].x + 20, self.panels["buttons"].y + 420 + i * 20))
        
        if self.show_help:
            self.draw_help_popup(screen)
    
    def draw_help_popup(self, screen):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        popup_width = 700
        popup_height = 500
        popup_x = (WIDTH - popup_width) // 2
        popup_y = (HEIGHT - popup_height) // 2
        pygame.draw.rect(screen, PANEL_BG, (popup_x, popup_y, popup_width, popup_height), border_radius=15)
        pygame.draw.rect(screen, PANEL_BORDER, (popup_x, popup_y, popup_width, popup_height), 3, border_radius=15)
        title = title_font.render("CONTROLS & INSTRUCTIONS", True, TEXT_COLOR)
        screen.blit(title, (popup_x + (popup_width - title.get_width()) // 2, popup_y + 20))
        close_button_rect = pygame.Rect(popup_x + popup_width - 40, popup_y + 10, 30, 30)
        pygame.draw.rect(screen, (200, 50, 50), close_button_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), close_button_rect, 2, border_radius=5)
        close_text = button_font.render("X", True, (255, 255, 255))
        screen.blit(close_text, (close_button_rect.x + 10, close_button_rect.y + 5))
        content_surface = pygame.Surface((popup_width - 40, 800), pygame.SRCALPHA)
        content_y = 0
        content_x = 20
        
        sections = [
            ("CUSTOMIZATION:", [
                "• Click on empty robot area to add sensors",
                "• Click on existing sensors to select them",
                "• Drag selected sensors to reposition",
                "• LEFT/RIGHT arrows to rotate selected sensor",
                "• DELETE or BACKSPACE to remove selected sensor",
                f"• Max sensor range: {SENSOR_MAX_RANGE:.1f} inches",
                "",
                "SENSOR COLORS:",
                "• Yellow: Normal sensor",
                "• Bright Yellow with Halo: Selected sensor",
                "• Red: Out of range (ignored)"
            ]),
            ("SIMULATION CONTROLS:", [
                "W/S - Move Forward/Backward",
                "A/D - Turn Left (CCW)/Right (CW)",
                "R - Randomize Robot Position",
                "K - Kidnap (teleport robot without resetting MCL)",
                "SPACE - Reset MCL Particles",
                "H - Toggle Tuning Panel"
            ]),
            ("COORDINATE SYSTEM:", [
                "• Field coordinates: 0° = UP, clockwise positive",
                "• Robot forward = 0° from robot perspective",
                "• Sensor angles relative to robot forward",
                "• Positive angles clockwise from robot forward"
            ])
        ]
        
        for section_title, items in sections:
            section_text = font.render(section_title, True, HIGHLIGHT)
            content_surface.blit(section_text, (content_x, content_y))
            content_y += 25
            for item in items:
                item_text = font.render(item, True, TEXT_DISABLED)
                content_surface.blit(item_text, (content_x + 10, content_y))
                content_y += 20
            content_y += 15
        
        self.help_content_height = content_y
        clip_rect = pygame.Rect(popup_x + 20, popup_y + 80, popup_width - 60, 400)
        screen.set_clip(clip_rect)
        screen.blit(content_surface, (popup_x + 20, popup_y + 80 + self.help_scroll_y))
        screen.set_clip(None)
        if self.help_content_height > 400:
            scrollbar_x = popup_x + popup_width - 30
            scrollbar_rect = pygame.Rect(scrollbar_x, popup_y + 80, 10, 400)
            pygame.draw.rect(screen, SCROLLBAR_COLOR, scrollbar_rect, border_radius=5)
            visible_height = 400
            content_height = self.help_content_height
            scroll_ratio = -self.help_scroll_y / max(1, content_height - visible_height)
            scrollbar_height = max(30, visible_height * visible_height / max(1, content_height))
            scrollbar_top = popup_y + 80 + scroll_ratio * (visible_height - scrollbar_height)
            handle_rect = pygame.Rect(scrollbar_x, scrollbar_top, 10, scrollbar_height)
            pygame.draw.rect(screen, SCROLLBAR_HANDLE, handle_rect, border_radius=5)
        
        hint_text = font.render("Press ESC or click X to close", True, TEXT_COLOR)
        screen.blit(hint_text, (popup_x + (popup_width - hint_text.get_width()) // 2, popup_y + popup_height - 30))

def world_to_screen(x, y):
    screen_x = FIELD_OFFSET_X + (x + FIELD_WIDTH/2) * SCALE
    screen_y = FIELD_OFFSET_Y + (FIELD_HEIGHT/2 - y) * SCALE
    return int(screen_x), int(screen_y)

class FieldScreen:
    def __init__(self, robot):
        self.field_map = FieldMap()
        self.robot = robot
        self.randomize_robot_position()
        
        self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta,
                      num_particles=1000, gaussian_stddev=1.5, gaussian_factor=1.0,
                      motion_noise_xy=0.15, motion_noise_theta=0.015)
        
        self.particles_slider = Slider(50, HEIGHT - 220, 250, 100, 3000, self.mcl.num_particles, "Particles", ":d")
        self.stdev_slider = Slider(50, HEIGHT - 180, 250, 0.1, 5.0, self.mcl.gaussian_stddev, "Gaussian StDev", ":.1f")
        self.factor_slider = Slider(50, HEIGHT - 140, 250, 0.1, 5.0, self.mcl.gaussian_factor, "Gaussian Factor", ":.1f")
        self.speed_slider = Slider(50, HEIGHT - 100, 250, 20.0, 80.0, robot.max_speed, "Drive Speed", ":.1f")
        self.angular_speed_slider = Slider(50, HEIGHT - 60, 250, 0.5, 10.0, robot.max_angular_speed, "Turn Speed", ":.1f")
        self.motion_noise_xy_slider = Slider(330, HEIGHT - 180, 250, 0.01, 0.5, self.mcl.motion_noise_xy, "Motion Noise XY", ":.2f")
        self.motion_noise_theta_slider = Slider(330, HEIGHT - 120, 250, 0.001, 0.1, self.mcl.motion_noise_theta, "Motion Noise Theta", ":.3f")
        
        self.menu_button = Button(WIDTH - 150, HEIGHT - 50, 120, 40, "MENU", (180, 100, 80))
        self.randomize_button = Button(WIDTH - 150, HEIGHT - 100, 120, 40, "RANDOMIZE", (100, 100, 180))
        self.kidnap_button = Button(WIDTH - 150, HEIGHT - 150, 120, 40, "KIDNAP", KIDNAP_COLOR)
        
        self.keys = {}
        self.show_tuning = True
    
    def randomize_robot_position(self):
        margin = 20
        self.robot.x = random.uniform(-FIELD_WIDTH/2 + margin, FIELD_WIDTH/2 - margin)
        self.robot.y = random.uniform(-FIELD_HEIGHT/2 + margin, FIELD_HEIGHT/2 - margin)
        self.robot.theta = random.uniform(0, 2 * math.pi)
    
    def kidnap_robot(self):
        margin = 20
        self.robot.x = random.uniform(-FIELD_WIDTH/2 + margin, FIELD_WIDTH/2 - margin)
        self.robot.y = random.uniform(-FIELD_HEIGHT/2 + margin, FIELD_HEIGHT/2 - margin)
        self.robot.theta = random.uniform(0, 2 * math.pi)
    
    def handle_event(self, event):
        if self.show_tuning:
            self.particles_slider.handle_event(event)
            self.stdev_slider.handle_event(event)
            self.factor_slider.handle_event(event)
            self.speed_slider.handle_event(event)
            self.angular_speed_slider.handle_event(event)
            self.motion_noise_xy_slider.handle_event(event)
            self.motion_noise_theta_slider.handle_event(event)
            self.mcl.num_particles = int(self.particles_slider.value)
            self.mcl.gaussian_stddev = self.stdev_slider.value
            self.mcl.gaussian_factor = self.factor_slider.value
            self.robot.max_speed = self.speed_slider.value
            self.robot.max_angular_speed = self.angular_speed_slider.value
            self.mcl.motion_noise_xy = self.motion_noise_xy_slider.value
            self.mcl.motion_noise_theta = self.motion_noise_theta_slider.value
        
        if self.menu_button.handle_event(event):
            return "menu"
        
        if self.randomize_button.handle_event(event):
            self.randomize_robot_position()
            self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta,
                          num_particles=self.mcl.num_particles,
                          gaussian_stddev=self.mcl.gaussian_stddev,
                          gaussian_factor=self.mcl.gaussian_factor,
                          motion_noise_xy=self.mcl.motion_noise_xy,
                          motion_noise_theta=self.mcl.motion_noise_theta)
        
        if self.kidnap_button.handle_event(event):
            self.kidnap_robot()
        
        if event.type == pygame.KEYDOWN:
            self.keys[event.key] = True
            if event.key == pygame.K_r:
                self.randomize_robot_position()
                self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta,
                              num_particles=self.mcl.num_particles,
                              gaussian_stddev=self.mcl.gaussian_stddev,
                              gaussian_factor=self.mcl.gaussian_factor,
                              motion_noise_xy=self.mcl.motion_noise_xy,
                              motion_noise_theta=self.mcl.motion_noise_theta)
            elif event.key == pygame.K_SPACE:
                self.mcl = MCL(self.field_map, self.robot.x, self.robot.y, self.robot.theta,
                              num_particles=self.mcl.num_particles,
                              gaussian_stddev=self.mcl.gaussian_stddev,
                              gaussian_factor=self.mcl.gaussian_factor,
                              motion_noise_xy=self.mcl.motion_noise_xy,
                              motion_noise_theta=self.mcl.motion_noise_theta)
            elif event.key == pygame.K_k:
                self.kidnap_robot()
            elif event.key == pygame.K_h:
                self.show_tuning = not self.show_tuning
        
        elif event.type == pygame.KEYUP:
            self.keys[event.key] = False
        
        return None
    
    def update(self, dt):
        self.robot.speed = 0
        self.robot.angular_speed = 0
        
        if self.keys.get(pygame.K_w):
            self.robot.speed = self.robot.max_speed
        if self.keys.get(pygame.K_s):
            self.robot.speed = -self.robot.max_speed
        if self.keys.get(pygame.K_a):
            self.robot.angular_speed = -self.robot.max_angular_speed
        if self.keys.get(pygame.K_d):
            self.robot.angular_speed = self.robot.max_angular_speed
        
        self.robot.update(dt)
        field_limit = FIELD_WIDTH/2 - WALL_THICKNESS - self.robot.width/2 - 2
        self.robot.x = max(-field_limit, min(field_limit, self.robot.x))
        self.robot.y = max(-field_limit, min(field_limit, self.robot.y))
        self.robot.update_sensors(self.field_map)
        dx, dy, dtheta = self.robot.get_deltas()
        self.mcl.run(self.robot.sensors, dx, dy, dtheta)
    
    def draw(self, screen):
        screen.fill(BACKGROUND)
        field_rect = pygame.Rect(FIELD_OFFSET_X, FIELD_OFFSET_Y, FIELD_WIDTH * SCALE, FIELD_HEIGHT * SCALE)
        if self.field_map.background_image:
            self.background_image = pygame.transform.flip(self.field_map.background_image, False, True)
            screen.blit(self.background_image, field_rect)
        else:
            pygame.draw.rect(screen, FIELD_BG, field_rect)
        pygame.draw.rect(screen, WALL_COLOR, field_rect, 3)
        for line in self.field_map.lines:
            start_x, start_y = world_to_screen(line[0][0], line[0][1])
            end_x, end_y = world_to_screen(line[1][0], line[1][1])
            pygame.draw.line(screen, OBSTACLE_COLOR, (start_x, start_y), (end_x, end_y), 3)
        self.draw_particles(screen)
        self.draw_robot(screen)
        self.draw_estimated_pose(screen)
        self.draw_ui(screen)
    
    def draw_particles(self, screen):
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
        corners = self.robot.get_corners()
        screen_corners = [world_to_screen(x, y) for x, y in corners]
        robot_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(robot_surface, ROBOT_COLOR, screen_corners)
        pygame.draw.polygon(robot_surface, ROBOT_OUTLINE, screen_corners, 2)
        center_x, center_y = world_to_screen(self.robot.x, self.robot.y)
        pygame.draw.circle(robot_surface, (255, 255, 255, 200), (center_x, center_y), 4)
        screen.blit(robot_surface, (0, 0))
        heading_length = 10
        heading_end_x = self.robot.x + heading_length * math.sin(self.robot.theta)
        heading_end_y = self.robot.y + heading_length * math.cos(self.robot.theta)
        heading_end_screen_x, heading_end_screen_y = world_to_screen(heading_end_x, heading_end_y)
        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (heading_end_screen_x, heading_end_screen_y), 3)
        for sensor in self.robot.sensors:
            sensor_pos = sensor.get_global_position(self.robot.x, self.robot.y, self.robot.theta)
            sensor_screen = world_to_screen(sensor_pos[0], sensor_pos[1])
            sensor_color = SENSOR_OUT_RANGE if not sensor.in_range else SENSOR_COLOR
            pygame.draw.circle(screen, sensor_color, sensor_screen, 6)
            sensor_angle = sensor.get_global_angle(self.robot.theta)
            end_x = sensor_pos[0] + sensor.distance * math.sin(sensor_angle)
            end_y = sensor_pos[1] + sensor.distance * math.cos(sensor_angle)
            end_screen = world_to_screen(end_x, end_y)
            ray_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            ray_color = RAY_OUT_RANGE if not sensor.in_range else RAY_COLOR
            pygame.draw.line(ray_surface, ray_color, sensor_screen, end_screen, 2)
            screen.blit(ray_surface, (0, 0))
            if sensor.in_range and sensor.distance < SENSOR_MAX_RANGE:
                pygame.draw.circle(screen, (255, 50, 50), end_screen, 4)
    
    def draw_estimated_pose(self, screen):
        if self.mcl.average_pose:
            x, y, theta = self.mcl.average_pose
            screen_x, screen_y = world_to_screen(x, y)
            pygame.draw.circle(screen, ESTIMATE_COLOR, (screen_x, screen_y), 10)
            pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 10, 2)
            heading_length = 10
            heading_end_x = x + heading_length * math.sin(theta)
            heading_end_y = y + heading_length * math.cos(theta)
            heading_end_screen_x, heading_end_screen_y = world_to_screen(heading_end_x, heading_end_y)
            pygame.draw.line(screen, ESTIMATE_COLOR, (screen_x, screen_y), (heading_end_screen_x, heading_end_screen_y), 4)
    
    def draw_ui(self, screen):
        title = title_font.render("FIELD SIMULATION", True, TEXT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        info_panel = pygame.Rect(20, 20, 320, 350)
        pygame.draw.rect(screen, PANEL_BG, info_panel, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, info_panel, 3, border_radius=10)
        info_y = 40
        info_lines = [
            "ROBOT INFO:",
            f"Position: ({self.robot.x:.1f}, {self.robot.y:.1f})",
            f"Heading: {math.degrees(self.robot.theta):.1f}°",
            f"Size: {self.robot.width:.1f} x {self.robot.length:.1f}",
            f"Sensors: {len(self.robot.sensors)}",
        ]
        for i, line in enumerate(info_lines):
            color = TEXT_COLOR if i == 0 else TEXT_DISABLED
            text = font.render(line, True, color)
            screen.blit(text, (40, info_y))
            info_y += 20
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
        info_y += 15
        mcl_lines = [
            "MCL INFO:",
            f"Particles: {len(self.mcl.particles)}",
            f"Used sensors: {sensors_in_range}/{len(self.robot.sensors)}",
            f"Estimate Error: {math.sqrt((self.robot.x - self.mcl.average_pose[0])**2 + (self.robot.y - self.mcl.average_pose[1])**2):.2f} in"
        ]
        for i, line in enumerate(mcl_lines):
            color = TEXT_COLOR if i == 0 else TEXT_DISABLED
            text = font.render(line, True, color)
            screen.blit(text, (40, info_y))
            info_y += 20
        legend_panel = pygame.Rect(20, 380, 320, 130)
        pygame.draw.rect(screen, PANEL_BG, legend_panel, border_radius=10)
        pygame.draw.rect(screen, PANEL_BORDER, legend_panel, 3, border_radius=10)
        legend_title = font.render("PARTICLE COLORS:", True, TEXT_COLOR)
        screen.blit(legend_title, (40, 395))
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
        if self.show_tuning:
            tuning_panel = pygame.Rect(20, HEIGHT - 280, 580, 260)
            pygame.draw.rect(screen, PANEL_BG, tuning_panel, border_radius=10)
            pygame.draw.rect(screen, PANEL_BORDER, tuning_panel, 3, border_radius=10)
            tuning_title = font.render("REAL-TIME TUNING (Press H to hide)", True, TEXT_COLOR)
            screen.blit(tuning_title, (50, HEIGHT - 265))
            self.particles_slider.draw(screen)
            self.stdev_slider.draw(screen)
            self.factor_slider.draw(screen)
            self.speed_slider.draw(screen)
            self.angular_speed_slider.draw(screen)
            self.motion_noise_xy_slider.draw(screen)
            self.motion_noise_theta_slider.draw(screen)
        else:
            tuning_title = font.render("Tuning hidden (Press H to show)", True, TEXT_DISABLED)
            screen.blit(tuning_title, (50, HEIGHT - 30))
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
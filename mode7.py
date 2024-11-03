import pygame as pg
import numpy as np
from settings import *
from numba import njit, prange

class spritesheet(object):
    def __init__(self, filename):
        self.sheet = pg.image.load(filename).convert()
    # Load a specific image from a specific rectangle
    def image_at(self, rectangle, colorkey = None):
        "Loads image from x,y,x+offset,y+offset"
        rect = pg.Rect(rectangle)
        image = pg.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey is -1:
                colorkey = image.get_at((0,0))
            image.set_colorkey(colorkey, pg.RLEACCEL)
        return image
    # Load a whole bunch of images and return them as a list
    def images_at(self, rects, colorkey = None):
        "Loads multiple images, supply a list of coordinates" 
        return [self.image_at(rect, colorkey) for rect in rects]
    # Load a whole strip of images
    def load_strip(self, rect, image_count, colorkey = None):
        "Loads a strip of images and returns them as a list"
        tups = [(rect[0]+rect[2]*x, rect[1], rect[2], rect[3])
                for x in range(image_count)]
        return self.images_at(tups, colorkey)


class Mode7:
    def __init__(self, app):
        self.app = app
        self.floor_tex = pg.image.load('textures/floor_2.png').convert()
        self.tex_size = self.floor_tex.get_size()
        self.floor_array = pg.surfarray.array3d(self.floor_tex)

        # Load the panoramic background image
        self.bg_tex = pg.image.load('textures/city.png').convert()
        self.bg_array = pg.surfarray.array3d(self.bg_tex)
        self.bg_size = self.bg_tex.get_size()

        self.screen_array = pg.surfarray.array3d(pg.Surface(WIN_RES))

        self.alt = 1.0
        self.angle = 0.0
        self.pos = np.array([0.0, 0.0])
        ss = spritesheet('toad.png')
        self.images = ss.images_at([(20+x*30,35,30,30) for x in range(1,11)],colorkey=(0, 0, 0))
        self.kart = self.images[0]

    def update(self):
        self.prev_angle = self.angle  # Save the previous angle before movement
        self.movement()
        
        # Calculate the turning rate
        turning_rate = self.angle - self.prev_angle
        normalized_turn = turning_rate / SPEED  # Normalize the turning rate
        
        # Map the normalized turn to an image index between 0 and 9
        image_index = int((normalized_turn + 1) * 4.5 + 0.5)
        image_index = max(0, min(10, image_index))  # Ensure index is within bounds
        
        # Update the kart image
        self.kart = self.images[image_index]
        
        # Render the frame
        self.screen_array = self.render_frame(
            self.floor_array, self.bg_array, self.screen_array,
            self.tex_size, self.bg_size, self.angle, self.pos, self.alt
        )

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

        # Blit the kart image onto the screen
        self.app.screen.blit(self.kart,(HALF_WIDTH-250,HALF_HEIGHT+300))

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def render_frame(floor_array, bg_array, screen_array, tex_size, bg_size, angle, player_pos, alt):

        sin, cos = np.sin(angle), np.cos(angle)
        FOV = np.pi / 2  # 90 degrees field of view

        for i in prange(WIDTH):
            new_alt = alt
            for j in range(HEIGHT):
                if j >= HALF_HEIGHT:
                    # Floor rendering
                    x = HALF_WIDTH - i
                    y = j + FOCAL_LEN
                    z = j - HALF_HEIGHT + new_alt

                    # Rotation
                    px = (x * cos - y * sin)
                    py = (x * sin + y * cos)

                    # Floor projection and transformation
                    floor_x = px / z - player_pos[1]
                    floor_y = py / z + player_pos[0]

                    # Floor position and color
                    floor_pos = (int(floor_x * SCALE % tex_size[0]),
                                 int(floor_y * SCALE % tex_size[1]))
                    # Check if floor texture coordinates are out of bounds
                    if (floor_x * SCALE > tex_size[0] or floor_y * SCALE > tex_size[1] or
                        floor_x * SCALE < 0 or floor_y * SCALE < 0):
                        # Compute background color at this pixel
                        angle_offset = (i / WIDTH - 0.5) * FOV
                        total_angle = angle + angle_offset
                        u = (total_angle % (2 * np.pi)) / (2 * np.pi) * bg_size[0]
                        v = (j / HEIGHT) * bg_size[1]
                        floor_col = bg_array[int(u % bg_size[0]), int(v % bg_size[1])]
                    else:
                        floor_col = floor_array[floor_pos]

                    depth = min(max(2.5 * (abs(z) / HALF_HEIGHT), 0), 1)
                    #fog = (1 - depth) * 230

                    #floor_col = (floor_col[0] * depth + fog,
                    #             floor_col[1] * depth + fog,
                    #             floor_col[2] * depth + fog)

                    # Fill screen array
                    screen_array[i, j] = floor_col

                    # Next depth
                    new_alt += alt
                else:
                    # Background rendering
                    angle_offset = (i / WIDTH - 0.5) * FOV
                    total_angle = angle + angle_offset
                    u = (total_angle % (2 * np.pi)) / (2 * np.pi) * bg_size[0]
                    v = (j / HEIGHT) * bg_size[1]
                    bg_col = bg_array[int(u % bg_size[0]), int(v % bg_size[1])]
                    screen_array[i, j] = bg_col

        return screen_array

    def movement(self):
        sin_a = np.sin(self.angle)
        cos_a = np.cos(self.angle)
        dx, dy = 0, 0
        speed_sin = SPEED * sin_a
        speed_cos = SPEED * cos_a

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            dx += speed_cos
            dy += speed_sin
        if keys[pg.K_s]:
            dx += -speed_cos
            dy += -speed_sin
        if keys[pg.K_a]:
            dx += speed_sin
            dy += -speed_cos
        if keys[pg.K_d]:
            dx += -speed_sin
            dy += speed_cos


        self.pos[0] += dx
        self.pos[1] += dy

        if keys[pg.K_LEFT]:
            self.angle -= SPEED
        if keys[pg.K_RIGHT]:
            self.angle += SPEED

        if keys[pg.K_q]:
            self.alt += SPEED
        if keys[pg.K_e]:
            self.alt -= SPEED
        self.alt = min(max(self.alt, 0.3), 4.0)

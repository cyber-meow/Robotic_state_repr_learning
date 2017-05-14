
import numpy as np
from environment.nav_env import NavEnv


# The egocentric view implementation

class NavEnvEgo(NavEnv):

    wall_colors = dict()
    wall_colors["front"] = np.array([0, 0, 0.5])  # dark blue
    wall_colors["left"] = np.array([0.4, 0.7, 1])  # light blue
    wall_colors["right"] = np.array([0, 0.7, 0])  # green
    wall_colors["back"] = np.array([0.2, 1, 0.6])  # palegreen
    top_color = np.array([0, 0, 0])  # black
    ground_color = np.array([0, 0.8, 0.8])  # aqua


    def observation(self):

        x, y = self.pos + np.array([0, 2])
        if y == 45:
            return np.tile(self.wall_colors["front"], (10, 10, 1))

        a1 = np.arctan(x/(45-y))  # angle in front, left
        a2 = np.arctan((45-x)/(45-y))  # angle in front, right
        b1 = np.pi/2 - a1
        b = np.arctan(y/x) + b1  # angle left
        c1 = np.pi/2 - a2
        c = np.arctan(y/(45-x)) + c1  # angle right

        img = np.empty((10, 10, 3))

        # left side of the vision field
        for i in range(5):
            start = i * np.pi / 6
            center = start + np.pi / 12
            end = start + np.pi / 6
            wall_dis = self._wall_dis(x, y, center, a1, b1, b)
            middle_color = self._middle_color(
                x, y, start, end, a1, b, self.wall_colors["left"])
            self._set_column_color(img, wall_dis, 4-i, middle_color)
            
        # right side of the vision field
        for i in range(5):
            start = i * np.pi / 6
            center = start + np.pi / 12
            end = start + np.pi / 6
            wall_dis = self._wall_dis(45-x, y, center, a2, c1, c)
            middle_color = self._middle_color(
                x, y, start, end, a2, c, self.wall_colors["right"])
            self._set_column_color(img, wall_dis, 5+i, middle_color)

        return img.reshape(-1)
                    

    @staticmethod
    def _wall_dis(hori_dis, y, angle, a1, b1, b):
        if angle <= a1:
            wall_dis = (45-y) / np.cos(angle)
        elif a1 < angle < a1 + b:
            wall_dis = hori_dis / np.cos(a1+b1-angle)
        else:
            wall_dis = y / np.cos(np.pi-angle)
        return wall_dis

    @classmethod
    def _middle_color(cls, x, y, start, end, a1, b, color):
        if end <= a1:
            middle_color = cls.wall_colors["front"]
        # must have end < a1+b
        elif start < a1 < end:
            middle_color = cls._mix_color(
                start, a1, end, cls.wall_colors["front"], color)
        elif a1 <= start and end <= a1+b:
            middle_color = color
        # must have a1 < start
        elif start < a1+b < end:
            middle_color = cls._mix_color(
                start, a1+b, end, color, cls.wall_colors["back"])
        # a1+b <= start
        else:
            middle_color = cls.wall_colors["back"]
        return middle_color


    @classmethod
    def _set_column_color(cls, img, wall_dis, col, middle_color):
        
        alpha = np.arctan(3/wall_dis)  # angle to the top
        beta = np.arctan(1/wall_dis)  # angle to the bottom
        
        # downside
        for i in range(7):
            start = i * np.pi / 18
            end = start + np.pi / 18
            if end <= beta:
                img[6-i, col] = middle_color
            elif start < beta < end:
                img[6-i, col] = cls._mix_color(
                    start, beta, end, middle_color, cls.ground_color)
            else:
                img[6-i, col] = cls.ground_color
        
        # upside
        for i in range(3):
            start = i * np.pi / 18
            end = start + np.pi / 18
            if end <= alpha:
                img[7+i, col] = middle_color
            elif start < alpha < end:
                img[7+i, col] = cls._mix_color(
                    start, alpha, end, middle_color, cls.top_color)
            else:
                img[7+i, col] = cls.top_color


    @staticmethod
    def _mix_color(left, mid, right, color1, color2):
        return ((mid-left)*color1 + (right-mid)*color2) / (right-left)


import numpy as np

from environment.nav_env_ego import NavEnvEgo
from utility import div0


class NavEnvExt(NavEnvEgo):
    """The Extended Navigation Task
    
    Per time step, the robot can choose to
    1. turn [-30, -15, 0, 15, 30] degrees
    2. and move fowards/backwards [-6, -3, 0, 3, 6] units
    All actions are subject to Gaussian noise with 0 mean and 10% std
    
    The distractors are:
    1. one rectangle on each wall of height 3 units and width 20 units
    2. three cycles on the ground of radius respectively 5, 7 and 10 units
    And they move randomly
    
    """
    
    # color settings

    eye_color = np.array([1, 1, 1])  # white
    
    rect_colors = {
        "front": np.array([1, 0, 1]),  # deep pink
        "left": np.array([1, 0.8, 1]),  # light pink
        "right": np.array([1, 0.5, 0]),  # orange
        "back": np.array([1, 1, 0.8]),  # light yellow
    }

    circle_color = np.array([1, 0.6, 0.8])  # pink


    def __init__(self):

        # orientation of the robot (-180° ~ 180° clockwise)
        self.orientation = 0
        # x,y coordinates of the robot
        self.pos = 2 * np.ones(2)

        rand = self._rand

        self.rect_pos = {
            "front": rand(10, 35),
            "left": rand(10, 35),
            "right": rand(10, 35),
            "back": rand(10, 35),
        }

        self.circles = {
            5: np.array([rand(5, 40), rand(5, 40)]),
            7: np.array([rand(7, 38), rand(7, 38)]),
            10: np.array([rand(10, 35), rand(10, 35)]),
        }


    @staticmethod
    def _rand(a, b): 
        return a + np.random.random()*(b-a)
    
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = pos
        self._pos_precompute()

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
        self._rad_orientation = orientation / 180 * np.pi

    @property
    def rad_orientation(self):
        return self._rad_orientation


    @property
    def actions(self):
        turn = np.arange(-30, 31, 15)
        mv = np.arange(-6, 7, 3)
        actions = np.dstack(np.meshgrid(turn, mv)).reshape(-1,2)
        return [list(a) for a in actions]

    @property
    def state(self):
        return np.append(self.pos, 
            [np.cos(self.rad_orientation), np.sin(self.rad_orientation)])

    def act(self, action):
        
        turn, mv = action[0], action[1]
        turn = np.random.normal(turn, abs(turn/10)) if turn else 0
        mv = np.random.normal(mv, abs(mv/10) if mv else 0)
        
        self.orientation += turn
        if self.orientation >= 180:
            self.orientation -= 360
        if self.orientation < -180:
            self.orientation += 360
        
        xmv = mv * np.sin(self.rad_orientation)
        ymv = mv * np.cos(self.rad_orientation)
        self.pos, bump = self._move_with_bound2(self.pos, (xmv, ymv), 2, 43)
        
        self.tick()
        
        if bump:
            reward = -1
        elif np.linalg.norm(45*np.ones(2) - self.pos) < 15:
            reward = 10
        else:
            reward = 0
        
        return self.observation(), reward
    

    @staticmethod
    def _move_with_bound(pos, dep, mi, ma):
        next_pos = pos + dep
        if next_pos < mi:
            return mi, True
        if next_pos > ma:
            return ma, True
        return next_pos, False

    def _move_with_bound2(self, pos, dep, mi, ma):
        next_xpos, bump_x = self._move_with_bound(pos[0], dep[0], mi, ma)
        next_ypos, bump_y = self._move_with_bound(pos[1], dep[1], mi, ma)
        return np.array([next_xpos, next_ypos]), bump_x and bump_y

    def tick(self):
        mvs = np.random.randn(4)
        for i, wall in enumerate(["front", "left", "right", "back"]):
            self.rect_pos[wall] = (
                self._move_with_bound(self.rect_pos[wall], mvs[i], 10, 35))[0]
        for r in self.circles:
            dep = np.random.randn(2)
            mi,ma = r,45-r
            self.circles[r] = (
                self._move_with_bound2(self.circles[r], dep, mi, ma))[0]


    def _pos_precompute(self):

        theta = self.rad_orientation
        # the position of the "eye"
        self._eyep = self.pos + 2 * np.array([np.sin(theta), np.cos(theta)])
        x,y = self._eyep
        
        self._a1 = np.arctan(div0(-x, 45-y))
        self._a2 = np.arctan(div0(45-x, 45-y))
        self._b = -np.pi/2 + np.arctan(div0(-y, x))
        self._c = np.pi/2 + np.arctan(div0(y, 45-x))

    
    def observation(self):
        img_large = self.egocentric_view(100)
        img = np.empty((10,10,3))
        for i in range(10):
            for j in range(10):
                img[i, j] = np.mean(
                    img_large[i*10:i*10+10, j*10:j*10+10], axis=(0,1))
        return img.reshape(-1)


    def egocentric_view(self, resolution):

        img_large = np.empty((resolution, resolution, 3))

        left_most = self.rad_orientation - np.pi*5/6
        right_most = self.rad_orientation + np.pi*5/6
        angles = np.linspace(left_most, right_most, resolution)
        angles[angles < -np.pi] += 2*np.pi
        angles[angles >= np.pi] -= 2*np.pi
        
        for i, angle in enumerate(angles):
            img_large[:,i] = self._column_color(angle, resolution)
        return img_large
    

    # angle in the case when self.orientation = 0
    def _column_color(self, angle, resolution):

        col_img = np.empty((resolution, 3))
        degrees = np.linspace(-7*np.pi/18, np.pi/6, resolution)
        wall_dis, wall_cord, wall_name = self._wall_information(angle)
        if wall_dis == 0:
            alpha = np.pi/2
            beta = -np.pi/2
        else:
            alpha = np.arctan(3/wall_dis)
            beta = np.arctan((-1)/wall_dis)
        
        if abs(wall_cord - self.rect_pos[wall_name]) <= 10:
            has_rectangle = True
            if wall_dis == 0:
                alpha1 = np.pi/2
                beta1 = -np.pi/2
            else:
                alpha1 = np.arctan(2.5/wall_dis)
                beta1 = np.arctan((-0.5)/wall_dis)
        else:
            has_rectangle = False

        x,y = self._eyep

        for i, degree in enumerate(degrees):
            if degree < beta:
                ground_dis = -1/np.tan(degree)
                look_point = np.array(
                    [x+ground_dis*np.sin(angle), y+ground_dis*np.cos(angle)])
                col_img[i] = self._ground_get_color(look_point)
            elif beta <= degree < alpha:
                if has_rectangle and beta1 <= degree < alpha1:
                    col_img[i] = self.rect_colors[wall_name]
                else:
                    col_img[i] = self.wall_colors[wall_name]
            else:
                col_img[i] = self.top_color

        return col_img


    def _wall_information(self, angle):

        x,y = self._eyep
        
        if self._a1 <= angle < self._a2:
            wall_dis = (45-y)/np.cos(angle)
            wall_cord = x + (45-y)*np.tan(angle)
            wall_name = "front"
        
        elif self._b <= angle < self._a1:
            angle += np.pi/2
            wall_dis = x/np.cos(angle)
            wall_cord = y + x*np.tan(angle)
            wall_name = "left"
        
        elif self._a2 <= angle < self._c:
            angle -= np.pi/2
            wall_dis = (45-x)/np.cos(angle)
            wall_cord = (45-y) + (45-x)*np.tan(angle)
            wall_name = "right"
        
        else:
            angle = angle + np.pi if angle < 0 else angle - np.pi
            wall_dis = y/np.cos(angle)
            wall_cord = (45-x) + y*np.tan(angle)
            wall_name = "back"
        
        return wall_dis, wall_cord, wall_name


    def _ground_get_color(self, cord):
        
        if np.linalg.norm(cord - self.pos) <= 2:
            xdis, ydis = cord - self.pos
            angle = np.arctan(np.divide(xdis, ydis))
            
            if ydis < 0:
                if xdis > 0:
                    angle += np.pi
                else:
                    angle -= np.pi
            
            angle_diff = min(
                abs(angle-self.rad_orientation), 
                abs(angle+2*np.pi-self.rad_orientation),
                abs(angle-2*np.pi-self.rad_orientation)
            )
            
            if angle_diff <= np.pi/18:
                return self.eye_color
            return self.bot_color
        
        for r in self.circles:
            circle_center = self.circles[r]
            if np.linalg.norm(cord - circle_center) <= r:
                return self.circle_color
        
        return self.ground_color


    def top_down_view(self, resolution):
        side = np.linspace(0, 45, resolution)
        cords = np.dstack(np.meshgrid(side, side))
        img = np.empty((resolution, resolution, 3))
        for i in range(resolution):
            for j in range(resolution):
                img[i,j] = self._ground_get_color(cords[i,j])
        return img


class NavEnvExtSpe(NavEnvExt):
    """
    The observation of the robot is the internal state, 
    used just for particular purposes (ex one curve in the ql experiment)
    """
    def observation(self):
        return self.state

    def show_observation(self, observation):
        return self.top_down_view(50)

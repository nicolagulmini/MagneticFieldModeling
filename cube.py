import numpy as np

class cube:
    
    def __init__(self, origin, side_length=4.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        self.interpolator = rbf_interpolator()
    
    def corners(self):
        return [self.origin_corner, 
                self.origin_corner + self.side_length * np.array([0., 0., 1.]),
                self.origin_corner + self.side_length * np.array([0., 1., 0.]),
                self.origin_corner + self.side_length * np.array([0., 1., 1.]),
                self.origin_corner + self.side_length * np.array([1., 0., 0.]),
                self.origin_corner + self.side_length * np.array([1., 0., 1.]),
                self.origin_corner + self.side_length * np.array([1., 1., 0.]),
                self.origin_corner + self.side_length * np.ones(3),
                ]
    
    def is_inside(self, point):
        # a 3-dim point
        if point[0] >= self.origin_corner[0] and point[1] >= self.origin_corner[1] and point[2] >= self.origin_corner[2]:
            if point[0] <= self.origin_corner[0] + self.side_length and point[1] <= self.origin_corner[1] + self.side_length and point[2] <= self.origin_corner[2] + self.side_length:
                return True
        return False
    
    def __str__(self):
        return "Cube of size %.1f x %.1f x %.1f, origin in (%.1f, %.1f, %.1f)" % (self.side_length, 
                                                                    self.side_length, 
                                                                    self.side_length, 
                                                                    self.origin_corner[0], 
                                                                    self.origin_corner[1], 
                                                                    self.origin_corner[2]) 

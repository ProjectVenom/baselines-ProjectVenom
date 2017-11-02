import cv2
from math import sin, cos, radians, degrees, atan, sqrt
import numpy as np


# t = target coordinates (numpy matrix) (x, y, z)
# c = camera coordinates (numpy matrix) (x, y, z)
# o = orientation vector (numpy matrix) (roll, pitch, yaw) as degrees
# w and h as width and height of the screen
# f as the focal length of the camera, None if the projection is to be done in airsim
def projection(t, c, o, w=1920.0, h=1080.0, f=None):
    z = float(t.item(0) - c.item(0))
    x = float(t.item(1) - c.item(1))
    y = float(t.item(2) - c.item(2))
    oZ = radians(o.item(0))
    oX = radians(o.item(1))
    oY = radians(o.item(2))

    oZ = 0.0

    dX = cos(oY) * (sin(oZ) * y + cos(oZ) * x) - sin(oY) * z
    dY = sin(oX) * (cos(oY) * z + sin(oY) * (sin(oZ) * y + cos(oZ) * x)) + cos(oX) * (cos(oZ) * y - sin(oZ) * x)
    dZ = cos(oX) * (cos(oY) * z + sin(oY) * (sin(oZ) * y + cos(oZ) * x)) - sin(oX) * (cos(oZ) * y - sin(oZ) * x)

    oZ = radians(-o.item(0))
    if f is None:    f = w / 2.0
    u = f * (dX / dZ)
    v = f * (dY / dZ)

    u_ = u * cos(oZ) - v * sin(oZ) + w / 2.0
    v_ = u * sin(oZ) + v * cos(oZ) + h / 2.0
    # (u2, v2) = projection(t, c, o, w, h, f)
    # if u2 != round(u_) or v2 != round(v_):
    #    print(c)
    #    print(o)
    target_in_front = test_closer(rot_mat(oX, oY, oZ), t, c)
    return (round(u_), round(v_)), target_in_front


def rot_mat(oX, oY, oZ):
    R_x = np.array([[1, 0, 0],
                    [0, cos(oX), -sin(oX)],
                    [0, sin(oX), cos(oX)]
                    ])

    R_y = np.array([[cos(oY), 0, sin(oY)],
                    [0, 1, 0],
                    [-sin(oY), 0, cos(oY)]
                    ])

    R_z = np.array([[cos(oZ), -sin(oZ), 0],
                    [sin(oZ), cos(oZ), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def test_closer(R, t, c):
    t0 = np.array([t.item(1), t.item(2), t.item(0)])
    c0 = np.array([c.item(1), c.item(2), c.item(0)])
    epsilon = 1e-4
    Rc = np.dot(c0, R)
    d0 = t0 - c0
    e = np.dot((d0 * epsilon), R)
    d1 = t0 - c0 - e
    d2 = t0 - c0 + e
    d0 = np.linalg.norm(d0)
    d1 = np.linalg.norm(d1)
    d2 = np.linalg.norm(d2)
    return d2 > d1


def get_o_from_pts(t, c):
    # t0 = np.array([t.item(1), t.item(2), t.item(0)])
    # c0 = np.array([c.item(1), c.item(2), c.item(0)])
    z = float(t.item(0) - c.item(0))
    x = float(t.item(1) - c.item(1))
    y = float(t.item(2) - c.item(2))

    '''
    x   y   z
    0   0   0
    0   0   1
    0   1   0
    0   1   1
    1   0   0
    1   0   1
    1   1   0
    1   1   1
    '''
    roll = 0
    if x == 0 and y == 0 and z == 0:
        return None
    elif x == 0 and y == 0 and z != 0:
        pitch = 0
        if z / abs(z) > 0:
            yaw = 0
        else:
            yaw = 180
    elif x == 0 and y != 0 and z == 0:
        yaw = 0
        pitch = y * 90 / abs(y)
    elif x == 0 and y != 0 and z != 0:
        if z / abs(z) > 0:
            yaw = 0
        else:
            yaw = 180
        pitch = atan(y / z)
    elif x != 0 and y == 0 and z == 0:
        yaw = x * 90 / abs(x)
        pitch = 0
    elif x != 0 and y == 0 and z != 0:
        yaw = atan(x / z)
        pitch = 0
    elif x != 0 and y != 0 and z == 0:
        yaw = 0
        pitch = atan(y / x)
    elif x != 0 and y != 0 and z != 0:
        yaw = atan((x / z))
        pitch = atan(sqrt(x ** 2 + z ** 2) / y)

    '''
    if z != 0: pitch = atan(sqrt(x**2+y**2)/z)
    else: pitch = 90
    if x != 0: yaw = atan((y/x))
    elif pitch == 90: yaw = 0
    else:
        if y == 0:
            yaw = 90
        else:
            yaw = 90*(y/abs(y))
    roll = 0
    '''

    return np.matrix([degrees(roll), degrees(pitch) - 90, degrees(yaw)])


def projection2(t, c, o, w=1920.0, h=1080.0, f=None):
    z = float(t.item(0) - c.item(0))
    x = float(t.item(1) - c.item(1))
    y = float(t.item(2) - c.item(2))
    oZ = radians(-o.item(0))
    oX = radians(-o.item(1))
    oY = radians(-o.item(2))

    # dist_coef = np.matrix([-0.000591, 0.000519, 0.000001, -0.000030, 0.000000])
    dist_coef = np.zeros(4)
    # rvec = i*j*k
    # rvec = np.eye(3)
    tvec = np.matrix([x, y, z])
    R = rot_mat(oX, oY, oZ)
    rvec, jacobian = cv2.Rodrigues(R)
    cameraMatrix = np.float64([[w / 2, 0, w / 2],
                               [0, w / 2, h / 2],
                               [0.0, 0.0, 1.0]])

    O_vec = np.zeros((1, 3))
    '''
    t_ = np.zeros((1,3))
    t_[0][0] = t.item(1)
    t_[0][1] = t.item(2)
    t_[0][2] = t.item(0)
    c_ = np.zeros((1,3))
    c_[0][0] = c.item(1)
    c_[0][1] = c.item(2)
    c_[0][2] = c.item(0)
    tvec = t_ - np.dot(c_,R)
    '''
    tvec = np.zeros((1, 3))
    tvec[0][0] = x
    tvec[0][1] = y
    tvec[0][2] = z
    # tvec = np.float64(t-c)
    pts = cv2.projectPoints(np.matrix([x, y, z]), rvec, O_vec, cameraMatrix, dist_coef)
    # print(pts[0])
    u = pts[0][0][0][0]
    v = pts[0][0][0][1]
    return (int(round(u)), int(round(v)))


t = np.matrix([-10.0, 10.0, -10.0])
c = np.matrix([[-24.02275109, 7.78893941, -9.58455081]])
# o = np.matrix([[ 172.63440558,   25.09867228, -147.62887133]])

# c = np.matrix([-21, 11, -11])

# c = np.matrix([-14.79488473,  -1.88013981, -13.47344467])
# o = np.matrix([89.97905282, 26.53989246, -90.09784977])
# print(projection2(t,c,o))
# o = np.matrix([o.item(0), o.item(1), o.item(2)])
'''
import cv2

objectPoints = t
tvec = t - c
rvec = np.zeros(3)
w = 256.0
h = 144.0
# print(pts)
o = get_o_from_pts(t, c)
print(o)
print(projection(t, c, o, h=144, w=256))
# print(projection2(t,c,o, h=144, w=256))
'''
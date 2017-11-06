from baselines.PythonClient import *
import time
hunter = AirSimClient(port=41451)
hunter.confirmConnection()
hunter.enableApiControl(True)
hunter.armDisarm(True)
# hunter.takeoff()
hunter.simSetPose(Vector3r(-20, 10, -10), hunter.toQuaternion(0, 0, 0))
hunter.moveByVelocity(0,0,0.1,0.1)
time.sleep(5)
for j in range(1):
    for i in range(50000):
        hunter.moveByVelocity(-5, 0, 0, 100)
        print((hunter.getVelocity().x_val,
                  hunter.getVelocity().y_val,
                  hunter.getVelocity().z_val))
    for i in range(50000):
        hunter.moveByVelocity(0, -5, 0, 100)
        print((hunter.getVelocity().x_val,
                  hunter.getVelocity().y_val,
                  hunter.getVelocity().z_val))
    for i in range(50000):
        hunter.moveByVelocity(5, 0, 0, 100)
        print((hunter.getVelocity().x_val,
                  hunter.getVelocity().y_val,
                  hunter.getVelocity().z_val))
    for i in range(50000):
        hunter.moveByVelocity(0, 5, 0, 100)
        print((hunter.getVelocity().x_val,
                  hunter.getVelocity().y_val,
                  hunter.getVelocity().z_val))
    for i in range(20000):
        hunter.moveByVelocity(0, 0, 0, 0.01)
        print((hunter.getVelocity().x_val,
                  hunter.getVelocity().y_val,
                  hunter.getVelocity().z_val))
    print((hunter.getPosition().x_val,
          hunter.getPosition().y_val,
          hunter.getPosition().z_val))
from picarx import Picarx
import time

if __name__ == "__main__":
    try:
        px = Picarx()

        px.set_dir_servo_angle(-20)
        time.sleep(0.5)
        px.forward(50)
        time.sleep(3)


        # px.set_dir_servo_angle(-20)
        # time.sleep(0.5)
        # px.set_dir_servo_angle(20)
        # time.sleep(0.5)
        # px.set_dir_servo_angle(-20)
        # time.sleep(0.5)
        # px.set_dir_servo_angle(0)

        # px.set_cam_tilt_angle(20)
        # time.sleep(0.5)
        # px.set_cam_tilt_angle(-20)
        # time.sleep(0.5)
        # px.set_cam_tilt_angle(0)
        # time.sleep(0.5)
        # px.forward(100)
        # time.sleep(1)
        # px.backward(100)
        # time.sleep(1)

    finally:
        px.stop()
        time.sleep(0.2)



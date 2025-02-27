from picarx import Picarx
import time
from time import sleep
from robot_hat import Music,TTS
import readchar

#distance to maintain from obstacles in front
safeDistance = 20

def main():

    flag_bgm = False
    music.music_set_volume(20)
    
    #constantly checkthe distance 
    while True:

        #read the obstacle distance in front of the car
        distance = round(px.ultrasonic.read(), 2)
        print("distance: ", distance)

        #check if the distance is liss than the safe distance
        if (distance < safeDistance):
            music.sound_play_threading('../sounds/car-double-horn.wav')
            sleep(0.05)

        else:
            print("Safe\n")
            
#run the main function
if __name__ == "__main__":
    main()



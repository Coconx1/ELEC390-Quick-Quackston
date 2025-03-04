from picarx import Picarx
import time
from time import sleep
from robot_hat import Music,TTS
import readchar

safeDistance = 20

def main():

    flag_bgm = False
    music.music_set_volume(20)
    #tts.lang("en-US")

    safeDistance = 20
    
    while True:
        distance = round(px.ultrasonic.read(), 2)
        print("distance: ", distance)
        if (distance < safeDistance):
            music.sound_play_threading('../sounds/car-double-horn.wav')
            sleep(0.05)

        else:
            print("Safe\n")
            
        




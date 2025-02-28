import RPi.GPIO as GPIO
import time

# Use BCM numbering for the GPIO pins
GPIO.setmode(GPIO.BCM)
# Setup GPIO pins as an output
GPIO.setup(18, GPIO.OUT) #Blinker L
GPIO.setup(18, GPIO.OUT) #Blinker R
GPIO.setup(18, GPIO.OUT) #Stop L
GPIO.setup(18, GPIO.OUT) #Stop R

try:
    while True:
        #For blinker
        GPIO.output(18, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(18, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second 
        GPIO.output(18, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(18, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second

try: 
    while True:
        #For stoping, this ASSSUMES WE NEED TO STOP THE CAR/SLOW DOWN FOR 5 SECONDS.
            #Run both pins, because one is physically wired. 
        GPIO.output(18, GPIO.HIGH) # Turn LED on
        time.sleep(5)              # Wait 5 second
        GPIO.output(18, GPIO.LOW)  # Turn LED off
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()                 # Clean up GPIO on exit

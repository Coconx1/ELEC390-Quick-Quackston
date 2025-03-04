import RPi.GPIO as GPIO
import time

# Use BCM numbering for the GPIO pins
GPIO.setmode(GPIO.BCM)
# Setup GPIO pins as an output
GPIO.setup(17, GPIO.OUT) #Blinker L
GPIO.setup(8, GPIO.OUT) #Blinker R
GPIO.setup(4, GPIO.OUT) #Stop L
GPIO.setup(27, GPIO.OUT) #Stop R

try:
    while True:
        #For blinker R
        GPIO.output(8, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(8, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second 
        GPIO.output(8, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(8, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second
try:
    while True:
        #For blinker L
        GPIO.output(17, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(17, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second 
        GPIO.output(17, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(17, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second
        
try: 
    while True:
        #For stoping, this ASSSUMES WE NEED TO STOP THE CAR/SLOW DOWN FOR 5 SECONDS.
            #Run both pins, because one not connected to PCB. 
        GPIO.output(4, GPIO.HIGH) # Turn LED on
        GPIO.output(27, GPIO.HIGH) # Turn LED on
        time.sleep(5)              # Wait 5 second
        GPIO.output(4, GPIO.LOW)  # Turn LED off
        GPIO.output(27, GPIO.LOW)  # Turn LED off

except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()                 # Clean up GPIO on exit

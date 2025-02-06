import RPi.GPIO as GPIO
import time

# Use BCM numbering for the GPIO pins
GPIO.setmode(GPIO.BCM)
# Setup GPIO pin 18 as an output
GPIO.setup(18, GPIO.OUT)

try:
    while True:
        GPIO.output(18, GPIO.HIGH) # Turn LED on
        time.sleep(1)              # Wait 1 second
        GPIO.output(18, GPIO.LOW)  # Turn LED off
        time.sleep(1)              # Wait 1 second
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()                 # Clean up GPIO on exit
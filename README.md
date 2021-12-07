# asthmaDetector
Device code for detecting asthma attacks in dogs using embedded audio classifier (Arduino Nano 33 BLE Sense)

detect_adc.cpp is the full working version of the code. detect_pdm.cpp is an older version that uses the Arduino Nano 33 BLE Sense's on board PDM microphone. detect_adc.cpp requires an electret microphone and a digital signal that goes high when a sound is detected. For this project we used the sound detector board from SparkFun (SEN-12642) which outputs both audio and a sound detection signal.

The device detects when a dog is having an ashtma attack and sends a notification via Bluetooth to a user's phone. Asthma detection is done using on-board audio classification. The audio classifier was built using Edge Impulse and employs a 2D convonultional neural network and MFE preprocessing. Inference time is roughly 120ms. To reduce power consumption, the device turns off if no sound has been detected within 10 seconds, and turns back on when a sound is detected.

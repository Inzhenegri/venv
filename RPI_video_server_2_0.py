# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from vidgear.gears import PiGear

import time
import os


def setup_gpio():
    os.system("sudo pigpiod")  # Launching GPIO library
    time.sleep(1)  # As i said it is too impatient and so if this delay is removed you will get an error
    import pigpio
    ESC = 17
    STEER = 18
    pi = pigpio.pi()
    pi.set_servo_pulsewidth(ESC, 0)
    pi.set_servo_pulsewidth(STEER, 0)
    time.sleep(1)
    # pi.set_servo_pulsewidth(ESC, 1500)
    # time.sleep(1)

    return pi,ESC,STEER

def calibrate(pi,ESC):   # Стандартная процедура автокалибровки для esc регулятора
    max_value = 2000  # Максимальное значение шим
    min_value = 700  # Минимальное значение шим
    pi.set_servo_pulsewidth(ESC, 0)
    print("Отключите питание (батарею) и нажмите Enter")
    inp = input()
    if inp == '':
        pi.set_servo_pulsewidth(ESC, max_value)
        print("Подключите батарею прямо сейчас. Вы должны услышать 2 звуквых сигнала. Затем дождитесь окончания сигнала и нажмите Enter")
        inp = input()
        if inp == '':
            pi.set_servo_pulsewidth(ESC, min_value)
            print ("Специальный сигнал скоро будет")
            time.sleep(7)
            print ("Ждите ....")
            time.sleep (5)
            print ("Не беспокойтесь, просто ждите.....")
            pi.set_servo_pulsewidth(ESC, 0)
            time.sleep(2)
            print ("Остановите ESC сейчас...")
            pi.set_servo_pulsewidth(ESC, min_value)
            time.sleep(1)
            print ("Калибровка завершена")
            # control() # You can change this to any other function you want
            pi.set_servo_pulsewidth(ESC, 1500)

def control(pi,ESC,speed,STEER,angle):
    pi.set_servo_pulsewidth(ESC, int(speed))
    pi.set_servo_pulsewidth(STEER,int(angle))

def stop(pi,ESC,connection): #This will stop every action your Pi is performing for ESC ofcourse.
    pi.set_servo_pulsewidth(ESC, 0)
    pi.stop()
    connection.close()

# add various Picamera tweak parameters to dictionary
options = {
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
    "CAP_PROP_FRAME_WIDTH":320,
    "CAP_PROP_FRAME_HEIGHT":240,

}

if __name__ == "__main__":
    pi, ESC, STEER = setup_gpio()
    control(pi, ESC, 1500, STEER, 1500)
    time.sleep(1)
    #calibrate(pi, ESC)
    # open pi video stream with defined parameters
    stream = VideoGear(source = 0, resolution = (100, 80), framerate = 60, **options).start()

    # activate Bidirectional mode
    options = {"bidirectional_mode": True}

    # Define NetGear server at given IP address and define parameters
    # !!! change following IP address '192.168.x.xxx' with client's IP address !!!
    server = NetGear(
        address="192.168.4.17",
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=True,
        max_retries = 1000000,
        **options


    # loop over until KeyBoard Interrupted
    while True:

        try:
            # read frames from stream
            frame = stream.read()

            # check for frame if Nonetype
            if frame is None:
                break

            # {do something with the frame here}

            # prepare data to be sent(a simple text in our case)
            target_data = "Hello, I am a Server."

            # send frame & data and also receive data from Client
            recv_data = server.send(frame, message=target_data)

            # print data just received from Client
            if not (recv_data is None):
                #print(recv_data)

                speed = recv_data[0]
                angle = recv_data[1]
                control(pi, ESC, speed, STEER, angle)
            else:
                speed = 1500
                angle = 1500
                control(pi, ESC, speed, STEER, angle)

        except KeyboardInterrupt:
            break

    # safely close video stream
    stream.stop()

    # safely close server
    server.close()
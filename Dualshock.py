#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file presents an interface for interacting with the Playstation 4 Controller
# in Python. Simply plug your PS4 controller into your computer using USB and run this
# script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4 controller.
#       if this is not the case, you will need to change the class accordingly.
#
# Copyright © 2015 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import pprint
import pygame
import pickle
import time
from threading import Thread
import argparse
import time
import os

speedoldmin = -2
speedoldmax = 2
speednewmin = 1000
speednewmax = 2000
speedoldrange = speedoldmax - speedoldmin
speednewrange = speednewmax - speednewmin

L2oldmin = -1
L2oldmax = 1
L2newmin = 1500
L2newmax = 1700
L2oldrange = L2oldmax - L2oldmin
L2newrange = L2newmax - L2newmin

L3oldmin = -1
L3oldmax = 1
L3newmin = 1700
L3newmax = 1250
L3oldrange = L3oldmax - L3oldmin
L3newrange = L3newmax - L3newmin
"""
R3oldmin = -1
R3oldmax = 1
R3newmin = 700
R3newmax = 1700
R3oldrange = R3oldmax-R3oldmin
R3newrange = R3newmax-R3newmin
R33oldmin = -1
R33oldmax = 1
R33newmin = 1100
R33newmax = 500
R33oldrange = R33oldmax-R33oldmin
R33newrange = R33newmax-R33newmin
"""

R33_init = 800

oldmins = -2
oldmaxs = 2
newmins = 1300
newmaxs = 1700
oldranges = oldmaxs - oldmins
newranges = newmaxs - newmins
T_f = 0.1
dt = 0.0
d_f = 0.0
dgas = 0
s_f = False
flag = True
angle = 0.0
speed = 0.0
R2 = 0
L2 = 0
L3 = 0


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

    return pi, ESC, STEER


def calibrate(pi, ESC):  # Стандартная процедура автокалибровки для esc регулятора
    max_value = 2000  # Максимальное значение шим
    min_value = 700  # Минимальное значение шим
    pi.set_servo_pulsewidth(ESC, 0)
    print("Отключите питание (батарею) и нажмите Enter")
    inp = input()
    if inp == '':
        pi.set_servo_pulsewidth(ESC, max_value)
        print(
            "Подключите батарею прямо сейчас. Вы должны услышать 2 звуквых сигнала. Затем дождитесь окончания сигнала и нажмите Enter")
        inp = input()
        if inp == '':
            pi.set_servo_pulsewidth(ESC, min_value)
            print("Специальный сигнал скоро будет")
            time.sleep(7)
            print("Ждите ....")
            time.sleep(5)
            print("Не беспокойтесь, просто ждите.....")
            pi.set_servo_pulsewidth(ESC, 0)
            time.sleep(2)
            print("Остановите ESC сейчас...")
            pi.set_servo_pulsewidth(ESC, min_value)
            time.sleep(1)
            print("Калибровка завершена")
            # control() # You can change this to any other function you want
            pi.set_servo_pulsewidth(ESC, 1500)


def control(pi, ESC, speed, STEER, angle):
    pi.set_servo_pulsewidth(ESC, int(speed))
    pi.set_servo_pulsewidth(STEER, int(angle))


def control_cam(pi, CAMUPDOWN, updown, LEFTRIGHT, leftright):
    pi.set_servo_pulsewidth(CAMUPDOWN, int(updown))
    pi.set_servo_pulsewidth(LEFTRIGHT, int(leftright))


def stop(pi, ESC, connection):  # This will stop every action your Pi is performing for ESC ofcourse.
    pi.set_servo_pulsewidth(ESC, 0)
    pi.stop()
    connection.close()


class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    controller = None
    axis_data = None
    button_data = None
    hat_data = None

    def init(self):
        """Initialize the joystick components"""

        # pygame.init()
        # pygame.joystick.quit()
        while True:
            pygame.init()
            self.check_connection = pygame.joystick.get_count()
            if self.check_connection == 1:
                break
            else:
                print(self.check_connection)
            pygame.joystick.quit()
            time.sleep(1)
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.axis_data = {}
        self.button_data = {}
        self.hat_data = {}

        """Listen for events to happen"""


def check_connection_const(delay):
    while True:
        pygame.init()
        check_connection = pygame.joystick.get_init()
        # if check_connection == 0:
        #    control(pi, ESC, 1500, STEER, 1700)
        #    print("Connection Lost")
        # else:
        #    pass
        print(check_connection)
        # pygame.joystick.quit()
        time.sleep(delay)


def main():
    # Ввод порта для передачи данных и флага калибровка. По умполчанию порт 1080, калибровка отключена

    # изменить порт можно командой:
    # -p <номер порта> пример: -p 1081

    # включить калибровку можно командой:
    # -с 1

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=False,
                    help="choose port: 1080 as default")
    ap.add_argument("-c", "--calibrate", required=False,
                    help="car motor calibration")
    args = vars(ap.parse_args())

    pi, ESC, STEER = setup_gpio()
    control(pi, ESC, 1500, STEER, 1500)
    time.sleep(1)
    if args["calibrate"] is not None:
        if int(args["calibrate"]) == 1:
            calibrate(pi, ESC)
        if int(args["calibrate"]) == 0:
            pass

    ps4 = PS4Controller()

    ps4.init()
    ps4.axis_data = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    ps4.button_data = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False,
                       9: False, 10: False, 11: False, 12: False, 13: False}
    ps4.hat_data = {}

    first_gas = 0
    s_f = False
    # t1 = Thread(target = check_connection_const, args = (1,))
    # t1.start()

    R3_init = 1305
    R33_init = 780
    R3 = 0
    R3_button = False
    L3_button = False
    x_button = False
    R33 = 0
    speed = 0
    mode = 0

    speedoldmin = -2
    speedoldmax = 2
    speednewmin = 1000
    speednewmax = 2000
    speedoldrange = speedoldmax - speedoldmin
    speednewrange = speednewmax - speednewmin
    while True:

        R3_init = R3_init + 0.8 * R3
        if R3_init > 1700:
            R3_init = 1700
            pass
        if R3_init < 500:
            R3_init = 500
            pass
        R33_init = R33_init - 0.8 * R33
        if R33_init > 2500:
            R33_init = 2500
            pass
        if R33_init < 500:
            R33_init = 500
            pass

        print(R3_init, R33_init)
        for event in pygame.event.get():
            st = time.time()
            if event.type == pygame.JOYAXISMOTION:
                ps4.axis_data[event.axis] = round(event.value, 2)
            elif event.type == pygame.JOYBUTTONDOWN:
                ps4.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                ps4.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                ps4.hat_data[event.hat] = event.value
            mode_flag = False
            R2 = ps4.axis_data[5]
            L3 = ps4.axis_data[0]
            L2 = ps4.axis_data[2]
            R3 = ps4.axis_data[4]
            R33 = ps4.axis_data[3]
            speed_oe = R2 - L2
            speed = (((speed_oe - speedoldmin) * speednewrange) / speedoldrange) + speednewmin
            L3 = (((L3 - L3oldmin) * L3newrange) / L3oldrange) + L3newmin
            # R3 = (((R3 - R3oldmin) * R3newrange) / R3oldrange) + R3newmin
            # R33 = (((R33 - R33oldmin) * R33newrange) / R33oldrange) + R33newmin

            x_button = ps4.button_data[0]
            if x_button == True:
                if mode_flag == False:
                    mode = mode + 1
                    if mode > 2:
                        mode = 0
                    pass

                    if mode == 0:
                        speedoldmin = -2
                        speedoldmax = 2
                        speednewmin = 1000
                        speednewmax = 2000
                        speedoldrange = speedoldmax - speedoldmin
                        speednewrange = speednewmax - speednewmin
                        speed = (((speed_oe - speedoldmin) * speednewrange) / speedoldrange) + speednewmin
                    elif mode == 1:
                        speedoldmin = -2
                        speedoldmax = 2
                        speednewmin = 1300
                        speednewmax = 1700
                        speedoldrange = speedoldmax - speedoldmin
                        speednewrange = speednewmax - speednewmin
                        speed = (((speed_oe - speedoldmin) * speednewrange) / speedoldrange) + speednewmin
                    elif mode == 2:
                        speedoldmin = -2
                        speedoldmax = 2
                        speednewmin = 1400
                        speednewmax = 1600
                        speedoldrange = speedoldmax - speedoldmin
                        speednewrange = speednewmax - speednewmin
                        speed = (((speed_oe - speedoldmin) * speednewrange) / speedoldrange) + speednewmin
                    mode_flag == True
                time.sleep(0.05)

            R3_button = ps4.button_data[12]
            if R3_button == True:
                R3_init = 1305
                R33_init = 780
            else:
                pass

            L3_button = ps4.button_data[11]
            if L3_button == True:
                R3_init = 1234
                R33_init = 2500
            else:
                pass

            control_cam(pi, 11, R3_init, 12, R33_init)
            if speed < 1501:
                if s_f == True:
                    # time.sleep(0.05)
                    control(pi, ESC, 1500, STEER, L3)
                    time.sleep(0.05)
                    control(pi, ESC, 1300, STEER, L3)
                    time.sleep(0.05)
                    control(pi, ESC, 1500, STEER, L3)
                    time.sleep(0.05)
                    s_f = False
                else:
                    pass
            else:
                s_f = True
            control(pi, ESC, speed, STEER, L3)


if __name__ == "__main__":
    main()
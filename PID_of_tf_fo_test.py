import math
import numpy as np
import time
import cv2 as cv

def PID(Input, Feedback, SatUp, SatDwn, Kp, Ti, Kd, Proportional =0, Differential=0, Integral=0, dt = 0):
    #start_time = time.time()
    Proportional = Kp*(Input-Feedback)
    if dt == 0:
        Differential = 0
    else:
        Differential = Kd*(Input-Feedback)/dt
    #dt = time.time() - start_time
    Integral += (Input-Feedback)*dt/Ti
    if Integral > SatUp:
        Integral = SatUp
    else:
        if Integral < SatDwn:
           Integral = SatDwn
    Output = Proportional + Differential + Integral
    if Output > SatUp:
          Output = SatUp
    else:
        if Output < SatDwn:
           Output = SatDwn 
    return Output, Proportional, Integral, Differential


#def TF_fo(dtime, input, output):
T = 0.1
K = 1
U = 0
V = 0
Load = 0
time_counter = 0
delta_time = 0
I = 0
W = 1000
size = W, W+900, 3
rook_image = np.zeros(size, dtype=np.uint8)
rook_window = "Drawing 1: Rook"
def my_line_dblue(img, start, end):
     thickness = 2
     line_type = 8
     cv.line(img,
              start,
              end,
              (255, 0, 0),
              thickness,
              line_type)
def my_line_cyan(img, start, end):
     thickness = 2
     line_type = 8
     cv.line(img,
              start,
              end,
              (255, 255, 0),
              thickness,
              line_type)
def my_line_yelw(img, start, end):
     thickness = 2
     line_type = 8
     cv.line(img,
              start,
              end,
              (0, 255, 255),
              thickness,
              line_type)
def my_line_gr(img, start, end):
     thickness = 2
     line_type = 8
     cv.line(img,
              start,
              end,
              (100, 255, 100),
              thickness,
              line_type)
def my_line_vlt(img, start, end):
     thickness = 2
     line_type = 8
     cv.line(img,
              start,
              end,
              (255, 50, 100),
              thickness,
              line_type)
while True:
    start_time = time.time()


    time_counter += delta_time
    V = ((U-V)*1/T*delta_time-Load)+V
      
    time.sleep(0.001)
    delta_time = time.time() - start_time
    if time_counter >= 1.3:
        Load = 0
    else:
        if time_counter>=0.7:
            Load = 0.01
        else:
            if time_counter>0:
                Load = 0
    U, P, I, D = PID(1, V, 2, -2, 1.5, 0.001, 0.04, Integral =I, dt = delta_time)
    my_line_dblue(rook_image, (int(900*time_counter), W-int(400*V)), (int(900*(time_counter+delta_time)), W-int(400*V)))
    #my_line_cyan(rook_image, (int(900*time_counter), W-int(400*U)), (int(900*(time_counter+delta_time)), W-int(400*U)))
    my_line_yelw(rook_image, (int(900*time_counter), W-int(400*I)), (int(900*(time_counter+delta_time)), W-int(400*I)))
    my_line_gr(rook_image, (int(900*time_counter), W-int(400*D)), (int(900*(time_counter+delta_time)), W-int(400*D)))
    my_line_vlt(rook_image, (int(900*time_counter), W-int(400*P)), (int(900*(time_counter+delta_time)), W-int(400*P)))
    print(V, time_counter)
    cv.imshow(rook_window, rook_image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    if time_counter > 2:
        break
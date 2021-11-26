# import required libraries
from vidgear.gears import NetGear
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from threading import Thread
import time
from multiprocessing import Process, current_process


def draw_axis(rook_image, zero_x, zero_y, W, width, numticks):
    cv2.line(rook_image, (int(width/numticks), 0),(int(width/numticks), W), color=(255, 255, 0), thickness=2)
    cv2.line(rook_image, (0, zero_y), (width, zero_y), color=(255, 255, 0), thickness=2)
    for i in range(numticks):
        cv2.line(rook_image, (int(i*width/numticks), 0), (int(i*width/numticks), W), color=(128, 128, 128), thickness=1)
        cv2.putText(rook_image, str(i-1), (int(i*width/numticks)+10, zero_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(rook_image, str(numticks-i-5), (zero_x+10, int(i*W/numticks)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)
        cv2.line(rook_image, (0, int(i*W/numticks)), (width, int(i*W/numticks)), color=(128, 128, 128), thickness=1)
def plot(size):

    time_counter = 0
    delta_time = 0
    current_pixel = 0
    prvs_pixel = 0
    W = size
    width = int(W*1.9)
    scale_x = 500
    scale_y = 0.1
    scale_u = 0.1
    size = W, width, 3
    zero_x = 60
    zero_y = int(W/2)
    zero_u = zero_y
    rook_image = np.zeros(size, dtype=np.uint8)
    rook_window = "Drawing 1: Rook{}".format(size)
    prvs_dvy_mapped = dvy_mapped
    prvs_dvy_f = dvy_f
    prvs_p = p
    prvs_i = i
    prvs_d = d
    prvs_u =u
    draw_axis(rook_image, zero_x, zero_y, W, width, numticks=10)
    while True:
        start_time = time.time()
        time_counter += delta_time
        if current_pixel >= width:
            rook_image = np.zeros(size, dtype=np.uint8)
            time_counter = 0.0
            current_pixel = 0
            prvs_pixel = 0
            draw_axis(rook_image, zero_x, zero_y, W, width, numticks=10)
        else:
            current_pixel = int(scale_x * time_counter)
            #cv2.line(rook_image, (prvs_pixel+zero_x, int(zero_y - scale_y * prvs_dvy_mapped)),
            #        (current_pixel+zero_x, int(zero_y - scale_y * dvy_mapped)), color=(0, 255, 0), thickness=1)
            cv2.line(rook_image, (prvs_pixel+zero_x, int(zero_y - scale_y * prvs_dvy_f)),
                    (current_pixel+zero_x, int(zero_y - scale_y * dvy_f)), color=(0, 0, 255), thickness=1)
            cv2.line(rook_image, (prvs_pixel + zero_x, int(zero_y - scale_y * prvs_p)),
                     (current_pixel + zero_x, int(zero_y - scale_y * p)), color=(255, 255, 0), thickness=1)
            cv2.line(rook_image, (prvs_pixel + zero_x, int(zero_y - scale_y * prvs_i)),
                     (current_pixel + zero_x, int(zero_y - scale_y * i)), color=(0, 255, 255), thickness=1)
            cv2.line(rook_image, (prvs_pixel + zero_x, int(zero_y - scale_y * prvs_d)),
                     (current_pixel + zero_x, int(zero_y - scale_y * d)), color=(0, 255, 0), thickness=1)
            cv2.line(rook_image, (prvs_pixel + zero_x, int(zero_u - scale_u * prvs_u)),
                     (current_pixel + zero_x, int(zero_u - scale_u * u)), color=(255, 255, 255), thickness=1)
            prvs_dvy_mapped = dvy_mapped
            prvs_dvy_f = dvy_f
            prvs_p = p
            prvs_i = i
            prvs_d = d
            prvs_u =u


            prvs_pixel = int(scale_x * (time_counter-delta_time))
        cv2.imshow(rook_window, rook_image)
        delta_time = time.time() - start_time
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


def my_line(img, start, end, thickness=2, line_type = 8):

    cv2.line(img,
            start,
            end,
            (0, 255, 0),
            thickness,
            line_type)


def my_line_red(img, start, end):
    thickness = 2
    line_type = 8
    cv2.line(img,
            start,
            end,
            (255, 255, 0),
            thickness,
            line_type)

def PID(Input, Feedback, SatUp, SatDwn, Kp, Ti, Kd,Integral=0, dt = 0.0):
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

def map(var, oldmin, oldmax, newmin, newmax):
    oldmin = oldmin
    oldmax = oldmax
    newmin = newmin
    newmax = newmax
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    mapped_value = (((var - oldmin) * newrange) / oldrange) + newmin
    return mapped_value

dvy_mapped = 0.0
dvy_f= 0.0
p = 0.0
i =0.0
d = 0.0
u=0.0
def stream():
    global dvy_mapped
    global dvy_f
    global p
    global i
    global d
    global u
    flag_first_flow_frame = False
    T_f = 0.1
    dvy_f = 0.0
    dt = 0.0

    Input =0.0
    i=0.0

    speed = 1500
    angle = 1500
    # activate Bidirectional mode
    options = {"bidirectional_mode": True}

    # Define NetGear Client at given IP address and define parameters
    # !!! change following IP address '192.168.x.xxx' with yours !!!
    client = NetGear(
        address="10.0.0.146",
        port="5454",
        protocol="tcp",
        pattern=1,
        receive_mode=True,
        logging=True,
        **options
    )
    while True:
        st = time.time()
        # prepare data to be sent
        target_data = [speed, angle]

        # receive data from server and also send our data
        data = client.recv(return_data=target_data)

        # check for data if None
        if data is None:
            break

        # extract server_data & frame from data
        server_data, frame = data

        # again check for frame if None
        if frame is None:
            break
        # if not (server_data is None):
        #     print(server_data)
        # {do something with the extracted frame and data here}
        #ROI = frame[160:240, 0:320].copy()
        ROI = frame[0:240, 0:320].copy()
        if not flag_first_flow_frame:
            prvs = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)

            flag_first_flow_frame = True
        pass

        next = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, 15, 1, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
        #dvx = -np.ma.average(flow[..., 0])
        dvy =  np.ma.average(flow[..., 1])
        #my_line(frame, (160, 120), (160 + int((500 * dvx) // 10), 120 + int((500 * dvy) // 10)))
        my_line(frame, (160, 120), (160, 120 + int((500 * dvy) // 10)))
        #my_line_red(frame, (160, 120), (160 + int((500 * dvx) // 10), 120))
        cv2.circle(frame, (160, 120), int((500 * abs(dvy)) // 10), (0, 255, 0), 2)
        #cv2.circle(frame, (160, 120), int((500 * abs(dvx)) // 10), (255, 255, 0), 2)
        prvs = next

        dvy_mapped = dvy*500

        dvy_f = ((dvy_mapped - dvy_f) * 1 / T_f * dt) + dvy_f









        u, p, i, d = PID(Input, Feedback=dvy_f, SatUp = 1000, SatDwn = 0, Kp = 0.2, Ti = 2.5, Kd = 0.001, Integral=i, dt=dt)
        speed = map(var=u, oldmin=0, oldmax=1000, newmin=1520, newmax=1700)
        dt = time.time() - st
        print(Input, speed)
        #frame = cv2.resize(ROI, (320, 240), interpolation=cv2.INTER_AREA)
        # let  print recieved server data


        # Show output window
        #cv2.imshow("Output Frame", frame)
        cv2.imshow("Output ROI", ROI)
        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            client.recv(return_data=[1500,1500])
            break
        if key == ord("a"):
            angle =1700
        if key == ord("d"):
            angle =1300
        if key == ord("s"):
            angle = 1500
        if key == ord(" "):
            Input = 0.0
        if key == ord("e"):
            Input +=10
        if key == ord("q"):
            Input -= 10




    # close output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()


if __name__ == "__main__":
    # thread2 = Thread(target=stream, args=())
    # thread2.start()

    time.sleep(2)
    thread1 = Thread(target=plot, args=(1000,))
    #thread1.start()
    stream()
    # thread2.join()
    #thread1.join()

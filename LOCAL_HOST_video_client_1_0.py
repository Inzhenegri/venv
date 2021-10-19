# import required libraries
from vidgear.gears import NetGear
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from threading import Thread
import time
from multiprocessing import Process, current_process
from scipy import signal

def plot_matplotlib():
    plt.ion()  ## Note this correction
    fig = plt.figure()
    plt.axis([0, 1000, -4000, 4000])
    plt.axis()
    i = 0
    x = list()
    y = list()
    while True:
        print(dvy_mapped)

        x.append(i)
        y.append(V)
        #plt.scatter(i, dvy_mapped, s=3, alpha=1);
        plt.plot(x, y, 'g', linewidth=2.0)
        i += 1
        plt.show()
        plt.pause(0.0001)
        if i>1000:

            fig.clear()
            plt.draw()
            x = list()
            y = list()
            i = 0.0
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
    scale_y = 0.15
    size = W, width, 3
    zero_x = 60
    zero_y = int(W/2)
    rook_image = np.zeros(size, dtype=np.uint8)
    rook_window = "Drawing 1: Rook{}".format(size)
    prvs_dvy_mapped = dvy_mapped
    prvs_dvy_f = dvy_f
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
            cv2.line(rook_image, (prvs_pixel+zero_x, int(zero_y - scale_y * prvs_dvy_mapped)),
                    (current_pixel+zero_x, int(zero_y - scale_y * dvy_mapped)), color=(0, 255, 0), thickness=1)
            cv2.line(rook_image, (prvs_pixel+zero_x, int(zero_y - scale_y * prvs_dvy_f)),
                    (current_pixel+zero_x, int(zero_y - scale_y * dvy_f)), color=(0, 0, 255), thickness=1)
            prvs_dvy_mapped = dvy_mapped
            prvs_dvy_f = dvy_f


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

dvy_mapped = 0.0
dvy_f= 0.0
def stream():
    global dvy_mapped
    global dvy_f
    flag_first_flow_frame = False
    T_f = 0.15
    dvy_f = 0.0
    dt = 0.0
    cap = cv2.VideoCapture(0)
    while True:
        st = time.time()
        ret, frame = cap.read()

        # {do something with the extracted frame and data here}
        #ROI = frame[160:240, 0:320].copy()
        ROI = frame[0:240, 0:320].copy()
        if not flag_first_flow_frame:
            prvs = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)

            flag_first_flow_frame = True
        pass

        next = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, 15, 1, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
        dvx = -np.ma.average(flow[..., 0])
        dvy =  np.ma.average(flow[..., 1])
        #my_line(frame, (160, 120), (160 + int((500 * dvx) // 10), 120 + int((500 * dvy) // 10)))
        my_line(frame, (160, 120), (160, 120 + int((500 * dvy) // 10)))
        #my_line_red(frame, (160, 120), (160 + int((500 * dvx) // 10), 120))
        cv2.circle(frame, (160, 120), int((500 * abs(dvy)) // 10), (0, 255, 0), 2)
        #cv2.circle(frame, (160, 120), int((500 * abs(dvx)) // 10), (255, 255, 0), 2)
        prvs = next

        dvy_mapped = dvy*500
        dt = time.time() - st
        dvy_f = ((dvy_mapped - dvy_f) * 1 / T_f * dt) + dvy_f

        #V = signal.TransferFunction
        #print(dvy_mapped)


        #frame = cv2.resize(ROI, (320, 240), interpolation=cv2.INTER_AREA)
        # let  print recieved server data


        # Show output window
        cv2.imshow("Output Frame", frame)
        cv2.imshow("Output ROI", ROI)
        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    # close output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()


if __name__ == "__main__":
    thread2 = Thread(target=stream, args=())
    thread2.start()
    time.sleep(2)
    thread1 = Thread(target=plot, args=(500,))
    thread1.start()



    thread2.join()
    thread1.join()

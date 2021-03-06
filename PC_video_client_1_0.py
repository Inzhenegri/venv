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
        plt.pause(0.001)
        if i>1000:

            fig.clear()
            plt.draw()
            x = list()
            y = list()
            i = 0.0
def filtered_flow():


    W = 1000
    size = W, int(W*1.9), 3
    rook_image2 = np.zeros(size, dtype=np.uint8)
    rook_window2 = "Drawing 2: Rook"
    prvs_V = dvy_mapped
    while True:
        #start_time = time.time()
        #time_counter +=delta_time
        time.sleep(0.01)


        U = 1
        # my_line(rook_image, (int(40 * time_counter), int(W/2) - int(0.05*dvy_mapped)),
        #         (int(40 * (time_counter + delta_time)), int(W/2) - int(0.05*dvy_mapped)))
        my_line(rook_image2, (int(40 * time_counter-delta_time), int(W/2) - int(0.05*prvs_V)),
                (int(40 * time_counter), int(W/2) - int(0.05*V)))
        prvs_V = dvy_mapped
        cv2.imshow(rook_window2, rook_image2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        if time_counter > 100:
            break
        #delta_time = time.time()-start_time
def plot():
    T = 0.1
    K = 1
    U = 0
    V = 0
    time_counter = 0
    delta_time = 0

    W = 340
    size = W, int(W*1.9), 3
    rook_image = np.zeros(size, dtype=np.uint8)
    rook_window = "Drawing 1: Rook"
    prvs_dvy_mapped = dvy_mapped
    while True:

        start_time = time.time()
        V = U * (delta_time / (T + delta_time)) + V * (T / (T + delta_time))
        time.sleep(0.01)
        delta_time = time.time() - start_time
        time_counter += delta_time
        U = 1
        # my_line(rook_image, (int(40 * time_counter), int(W/2) - int(0.05*dvy_mapped)),
        #         (int(40 * (time_counter + delta_time)), int(W/2) - int(0.05*dvy_mapped)))
        my_line(rook_image, (int(40 * time_counter-delta_time), int(W/2) - int(0.05*prvs_dvy_mapped)),
                (int(40 * time_counter), int(W/2) - int(0.05*dvy_mapped)))
        prvs_dvy_mapped = dvy_mapped
        cv2.imshow(rook_window, rook_image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        if time_counter > 100:
            break

def my_line(img, start, end):
    thickness = 2
    line_type = 8
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
V=0.0
delta_time = 0.0
time_counter =0.0
def stream():
    global dvy_mapped
    global V
    global time_counter
    global delta_time
    flag_first_flow_frame = False

    T_f=5
    K = 1
    U = 0

    #time_counter = 0
    #delta_time = 0

    # activate Bidirectional mode
    options = {"bidirectional_mode": True}

    # Define NetGear Client at given IP address and define parameters
    # !!! change following IP address '192.168.x.xxx' with yours !!!
    client = NetGear(
        #address="172.20.10.9",
        address="127.0.0.1",
        port="5454",
        protocol="tcp",
        pattern=1,
        receive_mode=True,
        logging=True,
        **options
    )
    # loop over
    while True:


        # prepare data to be sent
        target_data = "Hi, I am a Client here. Retry"

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
        #V = signal.TransferFunction
        #print(dvy_mapped)


        #frame = cv2.resize(ROI, (320, 240), interpolation=cv2.INTER_AREA)
        # let  print recieved server data
        if not (server_data is None):
            print(server_data)

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
    #thread1 = Thread(target=plot, args=())
    #thread1.start()
    #thread0 = Thread(target=filtered_flow(), args=())
    #thread0.start()


    thread2.join()
    #thread1.join()

    #thread0.join()
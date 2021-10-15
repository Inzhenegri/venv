# import required libraries
from vidgear.gears import NetGear
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from threading import Thread
import time

# def plot():
#     plt.ion()  ## Note this correction
#     fig = plt.figure()
#     plt.axis([0, 1000, -4000, 4000])
#     plt.axis()
#     i = 0
#     x = list()
#     y = list()
#     while True:
#         print(dvy_mapped)
#
#         x.append(i)
#         y.append(dvy_mapped)
#         #plt.scatter(i, dvy_mapped, s=3, alpha=1);
#         plt.plot(x, y, 'g', linewidth=2.0)
#         i += 1
#         plt.show()
#         plt.pause(0.001)
#         if i>1000:
#
#             fig.clear()
#             plt.draw()
#             x = list()
#             y = list()
#             i = 0.0

def my_line(img, start, end):
    thickness = 2
    line_type = 8
    cv.line(img,
            start,
            end,
            (255, 0, 0),
            thickness,
            line_type)
def plot(data):
    T = 0.1
    K = 1
    U = 0
    V = 0
    time_counter = 0
    delta_time = 0

    W = 420
    size = W, W, 3
    rook_image = np.zeros(size, dtype=np.uint8)
    rook_window = "Drawing 1: Rook"
    while True:
        start_time = time.time()
        V = U * (delta_time / (T + delta_time)) + V * (T / (T + delta_time))
        time.sleep(0.01)
        delta_time = time.time() - start_time
        time_counter += delta_time
        U = 1
        my_line(rook_image, (int(400 * time_counter), W - int(0.01*data)),
                (int(400 * (time_counter + delta_time)), W - int(0.01*data)))
        print(V, time_counter)
        cv2.imshow(rook_window, rook_image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        if time_counter > 1:
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
def stream():
    global dvy_mapped
    flag_first_flow_frame = False
    # activate Bidirectional mode
    options = {"bidirectional_mode": True}

    # Define NetGear Client at given IP address and define parameters
    # !!! change following IP address '192.168.x.xxx' with yours !!!
    client = NetGear(
        address="172.20.10.9",
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
    thread1 = Thread(target=plot, args=(dvy_mapped,))
    thread1.start()


    thread2.join()
    thread1.join()

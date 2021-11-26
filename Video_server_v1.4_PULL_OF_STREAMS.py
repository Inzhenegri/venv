import pickle
import time
import cv2 as cv
import multiprocessing
from multiprocessing import Pool
import socket
import numpy as np
import math

BUFFER_SIZE = 4096
img_size = [240, 360]
img_size_flow = [80, 80]
T = 0.1
T_f = 0.1
K = 0.01
U = 0
V = 0
Load = 0
time_counter = 0
delta_time = 0
I = 0
If = 0
main_p_g_i = 0
temp_delay = 0
speed=1600
dx = 0
speed_ref = 0
speed_ref_max = 0.0
oldmin = -0.42
oldmax = 0.42
newmin = 1250
newmax = 1700
oldrange = oldmax-oldmin
newrange = newmax-newmin

oldminf = -1
oldmaxf = 1
newminf = 1550
newmaxf = 1800
oldrangef = oldmaxf-oldminf
newrangef = newmaxf-newminf
prevcos =1.0
dvy_f = 0.0
flag = False
src = np.float32([[0, 240],
                  [360, 240],
                  [250, 150],
                  [80, 150]])

src_draw = np.array(src, dtype = np.int32)

dst = np.float32([[0, img_size[0]],
                  [img_size[1], img_size[0]],
                  [img_size[1], 0],
                  [0, 0]])
time_counter = 0


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
def send_cmd(cmd):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect the socket to the port where the CarControl is listening
    #server_address = ('localhost', 1081)
    server_address = ('192.168.4.1', 1090)
    sock.connect(server_address)
    try:
        # Send data
        message = cmd.encode()

        sock.sendall(message)
    finally:

        sock.close()

def PID(Input, Feedback, SatUp, SatDwn, Kp, Ti, Kd, Proportional = 0, Differential = 0, Integral=0, dt = 0.0):
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

def send_part(addr, port, control, x1, x2):
    st = time.time()
    sock = socket.socket()
    sock.connect((addr, port))
    data = pickle.dumps(control)
    sock.sendall(data)
    sock.close()
    dt = time.time() - st
    return dt

def recv_part(sock, buff_size):
    st = time.time()
    conn, addr = sock.accept()

    all_data = b''

    while True:
        data = conn.recv(buff_size)
        if not data:
            break
        all_data += data
    dt = time.time() - st
    part = pickle.loads(all_data)

    conn.close()
    return part, dt

class sock(object):
    def __init__(self, addr='', port=50007):
        self.sock = socket.socket()
        self.addr = addr
        self.port = port
    def sock_setup(self):
        self.sock.bind((self.addr, self.port))
        self.sock.listen()
        print('Sock name: {}'.format(self.sock.getsockname()))



if __name__ == '__main__':
    multiprocessing.freeze_support()
    sock1 = sock('192.168.4.218', 50008)
    sock1.sock_setup()

    pool = Pool(processes=1)
    result = pool.apply_async(recv_part, (sock1.sock, BUFFER_SIZE))
    flw_1, dt = result.get()
    frame_prv = cv.resize(flw_1, (img_size_flow[1], img_size_flow[0]))
    prv = cv.cvtColor(frame_prv, cv.COLOR_BGR2GRAY)
    while True:
        st= time.time()
        result = pool.apply_async(recv_part, (sock1.sock, BUFFER_SIZE))
        part1, dt = result.get()
        #frame = cv.resize(part1, (img_size_flow[1], img_size_flow[0]))
        frame2 = cv.resize(part1, (img_size_flow[1], img_size_flow[0]))
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prv, next, None, 0.5, 1, 15, 1, 5, 1.2, 0)
        dvx = -np.ma.average(flow[..., 0])
        dvy = np.ma.average(flow[..., 1])
        prv = next
        resized = cv.resize(part1, (img_size[1], img_size[0]))
        dvy_f = ((dvy - dvy_f) * 1 / T_f * dt) + dvy_f
        #cv.imshow("r_channel", next)

        r_channel = resized[:, :, 2]
        binary = np.zeros_like(r_channel)
        binary[(r_channel > 200)] = 1
#        cv.imshow("r_channel", binary)

        hls = cv.cvtColor(resized, cv.COLOR_BGR2HLS)
        s_channel = resized[:, :, 2]
        binary2 = np.zeros_like(s_channel)
        binary2[(s_channel > 160)] = 1
        # cv.imshow("s_channel", binary2)

        allBinary = np.zeros_like(binary)
        allBinary[((binary == 1) | (binary2 == 1))] = 255
        # cv.imshow("binary", allBinary)

        allBinary_visual = allBinary.copy()
        cv.polylines(allBinary_visual, [src_draw], True, 255)
        cv.imshow("binary", allBinary_visual)

        M = cv.getPerspectiveTransform(src, dst)
        warped = cv.warpPerspective(allBinary, M, (img_size[1], img_size[0]), flags=cv.INTER_LINEAR)
        # cv.imshow("warped", warped)
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        pointleft = midpoint - 20
        pointright = midpoint + 20
        IndWitestColumnL = np.argmax(histogram[:pointleft])
        IndWitestColumnR = np.argmax(histogram[pointright:]) + pointright
        warped_visual = warped.copy()
        #    cv.line(warped_visual, (IndWitestColumnL, 0), (IndWitestColumnL, warped_visual.shape[0]), 110, 2)
        #    cv.line(warped_visual, (IndWitestColumnR, 0), (IndWitestColumnR, warped_visual.shape[0]), 110, 2)
        # cv.imshow("WitestColumn", warped_visual)

        nwindows = 10
        window_height = np.int(warped.shape[0] / nwindows)
        window_half_width = 25

        XCenterLeftWindow = IndWitestColumnL
        XCenterRightWindow = IndWitestColumnR

        left_lane_inds = np.array([], dtype=np.int16)
        right_lane_inds = np.array([], dtype=np.int16)

        out_img = np.dstack((warped, warped, warped))

        nonzero = warped.nonzero()
        WhitePixelIndY = np.array(nonzero[0])
        WhitePixelIndX = np.array(nonzero[1])

        for window in range(nwindows):

            win_y1 = warped.shape[0] - (window + 1) * window_height
            win_y2 = warped.shape[0] - (window) * window_height
            #
            left_win_x1 = XCenterLeftWindow - window_half_width
            left_win_x2 = XCenterLeftWindow + window_half_width
            right_win_x1 = XCenterRightWindow - window_half_width
            right_win_x2 = XCenterRightWindow + window_half_width

            #cv.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50+window*21,0,0),2)
            #cv.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0,0,50+window*21),2)
            # cv.imshow("windows", out_img)

            good_left_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                              (WhitePixelIndX >= left_win_x1) & (WhitePixelIndX <= left_win_x2)).nonzero()[0]

            good_right_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                               (WhitePixelIndX >= right_win_x1) & (WhitePixelIndX <= right_win_x2)).nonzero()[0]

            left_lane_inds = np.concatenate((left_lane_inds, good_left_inds))
            right_lane_inds = np.concatenate((right_lane_inds, good_right_inds))

            if len(good_left_inds) > 50:
                XCenterLeftWindow = np.int(np.mean(WhitePixelIndX[good_left_inds]))
            if len(good_right_inds) > 50:
                XCenterRightWindow = np.int(np.mean(WhitePixelIndX[good_right_inds]))

        out_img[WhitePixelIndY[left_lane_inds], WhitePixelIndX[left_lane_inds]] = [255, 0, 0]
        out_img[WhitePixelIndY[right_lane_inds], WhitePixelIndX[right_lane_inds]] = [0, 0, 255]
#       cv.imshow("Lane", out_img)

        leftx = WhitePixelIndX[left_lane_inds]
        lefty = WhitePixelIndY[left_lane_inds]

        rightx = WhitePixelIndX[right_lane_inds]
        righty = WhitePixelIndY[right_lane_inds]

        if  np.sum(leftx, 0) == 0 | np.sum(lefty, 0) == 0:
            print("empty left")
            main_high_x = midpoint + 50
        else:
            if np.sum(rightx, 0) == 0 | np.sum(righty, 0) == 0:
                print("empty right")
                main_high_x = midpoint - 50
            else:
                right_fit = np.polyfit(righty, rightx, 2)
                left_fit = np.polyfit(lefty, leftx, 2)
                center_fit = ((left_fit + right_fit) / 2)
                for ver_ind in range(out_img.shape[0]):
                    gor_ind = ((center_fit[0]) * (ver_ind ** 2) +
                               center_fit[1] * ver_ind +
                               center_fit[2])
         #           #        cv.circle(out_img, (int(gor_ind),int(ver_ind)),2,(255,0,266),1)
                    if ver_ind == int(out_img.shape[0] // 1.1):
                        main_low_x = gor_ind
                        main_low_y = ver_ind
                        pass
                    if ver_ind == int(out_img.shape[0] // 1.5):
                        main_high_x = gor_ind
                        main_high_y = ver_ind
                        pass
          #          cv.circle(out_img, (int(main_p_g_i),int(ver_ind)),5,(255,50,100),5)
         #           cv.imshow("CenterLine", out_img)
                dx = -(midpoint - main_low_x)


        dt = time.time() - st
        angle_rad = -math.atan((main_high_x-(midpoint+40))/main_high_y)
        cos = math.cos(angle_rad)
        K_cos = prevcos/cos
        prevcos = cos
        speed_ref += 0.002
        if speed_ref > speed_ref_max:
            speed_ref = speed_ref_max
        #U, P, I, D = PID(0, dx, 1, -1, 0.012, 0.2, 0.001, Integral=I, dt=dt)
        #K_alpha = (3.14/6)/1
        #b = U*K_alpha
        #cos_b = math.cos(b)
        #v_fdbck = float(dvy_f)/cos_b
        Uf, Pf, If, Df = PID(speed_ref, dvy_f, 1, -1, 0.02, 7, 0.001, Integral=If, dt=dt)

        angle = (((angle_rad - oldmin) * newrange) / oldrange) + newmin
        speed = (((Uf - oldminf) * newrangef) / oldrangef) + newminf
        #        address = '192.168.4.1'
        #        portrpi = 50008
        #        wrkr2 = pool.apply_async(send_part, (address, portrpi, control, 0, 329))
        #        dt2 = wrkr2.get()


        key = cv.waitKey(1)

        #if key == ord('q'):
        #    send_cmd(DEFAULT_CMD)
        #    break
        if key == ord('s'):
            speed_ref_max = speed_ref_max - 0.05
            pass
        if key == ord('w'):
            speed_ref_max = speed_ref_max + 0.05
            pass
        if key == ord('e'):
            speed_ref_max = 0
            pass
        if key == ord('q'):
            speed = 1500
            send_cmd('00/' + str(speed) + '/' + str(angle))
            time.sleep(0.05)
            print('stopped')
            break

            pass
        send_cmd('00/' + str(speed) + '/' + str(angle))
        #angle = (((U - oldmin) * newrange) / oldrange) + newmin

        print(dvy_f, dt, speed_ref_max, speed_ref)
        #time_counter += 200*dt
        #print(time_counter)
        #my_line_dblue(rook_image, ((int(10000 * float(dvy))), int(1 * time_counter)), ((int(10000 * float(dvy)), int(1 * (time_counter + 1)) )))
        #cv.imshow(rook_window, rook_image)


        #my_line_dblue(rook_image, (10, 50), (200, 200))
        #my_line_dblue(rook_image, (int(1 * time_counter), int(W/2)-int(1000 * float(dvy))), (int(1 * (time_counter + 5)), int(W/2)-int(1000 * float(dvy))))
        #my_line_cyan(rook_image, (int(1 * time_counter), int(W/2)-int(1000 * float(dvy_f))), (int(1 * (time_counter + 5)), int(W/2)-int(1000 * float(dvy_f))))
        #if time_counter>1000:
        #    rook_image = np.zeros(size, dtype=np.uint8)
        #    cv.imshow(rook_window, rook_image)
        #    passrook_image = np.zeros(size, dtype=np.uint8)
        #    time_counter = 0
        #cv.imshow(rook_window, rook_image)
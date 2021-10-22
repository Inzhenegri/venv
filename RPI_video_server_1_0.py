# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from vidgear.gears import PiGear

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

# open pi video stream with defined parameters
stream = VideoGear(source = 0, resolution = (100, 80), framerate = 60, **options).start()

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# Define NetGear server at given IP address and define parameters
# !!! change following IP address '192.168.x.xxx' with client's IP address !!!
server = NetGear(
    address="10.10.0.14",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
    max_retries = 1000000,
    **options
)  #

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
            print(recv_data)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
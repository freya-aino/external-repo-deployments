import traceback
import time
import imagezmq
import cv2
import os
import facer


image_reciever = imagezmq.ImageHub(os.environ["ADDRESS"], REQ_REP = False)
image_reciever.connect(os.environ["ADDRESS"])


try:

    while True:

        dt = time.perf_counter()

        information, frame = image_reciever.recv_image()

        print("recieved_fps=", 1 / (time.perf_counter() - dt))








        cv2.imshow("a", frame)
        cv2.waitKey(1)

except:
    traceback.print_exc()
finally:
    image_reciever.close()


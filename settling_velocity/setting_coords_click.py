import numpy as np
import cv2
from matplotlib import pyplot as plt


def convert_to_mp4(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file)

def click_event(event, x, y, flags, params):
    global frameNo, data_list, img

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        data_list.append((frameNo, x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        data_list.append((frameNo, x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.imshow('image', img)

def play_video(filename):
    global frameNo, data_list, img

    vidcap = cv2.VideoCapture(filename)
    frameNo = 0
    data_list = []

    while True:
        success, img = vidcap.read()
        if not success:
            break

        font_a = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Frame: ' + str(frameNo), (10, 30), font_a, 1, (0, 255, 0), 2)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 2000, 1000)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('z'):
            frameNo += 1
            

    cv2.destroyAllWindows()
    data_list = np.asarray(data_list)
    folder = "settling velocities/05_ps_results"
    np.savetxt(folder + "//" + filename_save + ".txt", data_list.astype(int), fmt='%i', delimiter=";")
    print("number of points", len(data_list[:, 1]))
    print("all data saved")
    plt.scatter(data_list[:, 0], data_list[:, 2])

filename_vid = 'settling velocities/05_ps_mp4/05_cup_070.mp4'
filename_save = filename_vid[30:40]
play_video(filename_vid)



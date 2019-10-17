import numpy as np
import cv2
from matplotlib import pyplot as plt

#Get Video
video = cv2.VideoCapture("Your file name here")
ret, frame = video.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
frame_gauss = cv2.GaussianBlur(frame_gray, (5, 5), 0)

#Capture ROI area and calculate points
roi = cv2.selectROI(frame)
x1 = roi[0]             #left
x2 = roi[0] + roi[2]    #right
y1 = roi[1]             #upper
y2 = roi[1] + roi[3]    #under

roi_frame = frame_gauss[y1:y2, x1:x2]

#Calulate Histogram of ROI area and show Histogram
hist = cv2.calcHist(images = [roi_frame], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
hist = hist.flatten()
plt.title('hist')
plt.plot(hist, color = 'r')
binX = np.arange(256)
plt.bar(binX, hist, width = 1, color = 'b')
plt.show()

#Find local maximum value
#Lane area has white color so we find maximum white value in ROI area
maximum = 0
idx = 0
check = True

for i, v in reversed(list(enumerate(hist))):
    if v >= maximum:
        if check:
            maximum = v
            idx = i
        else:
            break
    else:
        check = False

print('Local Minimum : idx = {}'.format(idx))

#To write mask and result file
file_num = 1

#Read each frame in video and find lane
#If you want write file remove "#"
while True:
    ret, frame = video.read()
    filename = str(file_num) + ".png"
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gauss = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_roi = frame_gray[y1:y2, x1:x2]
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8)
    mask[y1:y2, x1:x2] = frame_roi
    mask_white = cv2.inRange(mask, idx * 2 - 255, 255)
    result = cv2.bitwise_and(frame_gauss, mask_white)
    #cv2.imwrite("./mask/" + filename, result)          
    result = cv2.Canny(result, 50, 150)
    
    lines = cv2.HoughLinesP(result, 2, np.pi/180, 20, np.array([]), minLineLength = 20, maxLineGap = 50)
    line_img = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x3, y3, x4, y4 in line:
                cv2.line(frame, (x3, y3), (x4, y4), [255, 0, 0], 3)

    #cv2.imwrite("./result/" + filename, frame)
    cv2.imshow("result", frame)
    file_num += 1

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()

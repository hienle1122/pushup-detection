import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("AiTrainer/joseph.mov")

detector = pm.poseDetector()
left_count = 0
left_dir = 0
right_count = 0
right_dir = 0
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("AiTrainer/test.jpg")
    img = detector.findPose(img, True)
    lmList = detector.findPosition(img, True)
    if len(lmList) != 0:
        # Right Arm
        right_angle = detector.findAngle(img, 12, 14, 16)
        right_per = 100-np.interp(right_angle, (100, 160), (0, 100))
        right_bar = np.interp(right_angle, (100, 160), (100, 650))

        # Left Arm
        left_angle = detector.findAngle(img, 11, 13, 15)
        left_per = np.interp(left_angle, (195, 250), (0, 100))
        left_bar = np.interp(left_angle, (195, 250), (650, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if left_per == 100:
            color = (0, 255, 0)
            if left_dir == 0:
                left_count += 0.5
                left_dir = 1
        if left_per == 0:
            color = (0, 255, 0)
            if left_dir == 1:
                left_count += 0.5
                left_dir = 0

        if right_per == 100:
            color = (0, 255, 0)
            if right_dir == 0:
                right_count += 0.5
                right_dir = 1
        if right_per == 0:
            color = (0, 255, 0)
            if right_dir == 1:
                right_count += 0.5
                right_dir = 0

        # Draw Right Bar
        cv2.rectangle(img, (1110, 100), (1135, 650), color, 3)
        cv2.rectangle(img, (1110, int(right_bar)),
                      (1135, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(right_per)} %', (1050, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw left Pushup Count
        cv2.putText(img, "Left Count: " + str(int(left_count)), (45, 700), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # Draw Right Bar
        cv2.rectangle(img, (100, 100), (125, 650), color, 3)
        cv2.rectangle(img, (100, int(left_bar)),
                      (125, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(left_per)} %', (50, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Right Pushup Count
        cv2.putText(img, "Right Count: " + str(int(right_count)), (800, 700), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (600, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

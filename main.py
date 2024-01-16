from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8l-pose.pt")
# results = model.track('example_input1.jpg', show=True, device='0')
# test image
# for r in results:
#     image = r.orig_img
#     keypoints = r.keypoints.data.tolist()
#     frame_with_box = r.plot()
#     boxes = r.boxes.data.tolist()
#     kp = 0
#     for box in boxes:
#         print("person id:", int(box[4]))
#         vai_phai_y = (keypoints[kp][6])
#         canhtay_phai = keypoints[kp][8]
#         co_tay_phai = (keypoints[kp][10])

#         vai_trai_y = (keypoints[kp][5])
#         canhtay_trai = keypoints[kp][7]
#         co_tay_trai = (keypoints[kp][9])

#         image = cv2.circle(image, (int(vai_trai_y[0]), int(
#             vai_trai_y[1])), radius=5, color=(0, 0, 255), thickness=-1)  # vaitrai
#         image = cv2.circle(image, (int(co_tay_trai[0]), int(
#             co_tay_trai[1])), radius=5, color=(0, 0, 255), thickness=-1)  # vaitrai
#         image = cv2.circle(image, (int(canhtay_trai[0]), int(
#             canhtay_trai[1])), radius=5, color=(0, 0, 255), thickness=-1)  # canhtay_trai
#         image = cv2.circle(image, (int(canhtay_phai[0]), int(
#             canhtay_phai[1])), radius=5, color=(0, 0, 255), thickness=-1)  # canhtay_phai
#         image = cv2.circle(image, (int(vai_phai_y[0]), int(
#             vai_phai_y[1])), radius=5, color=(0, 0, 255), thickness=-1)  # vaitrai
#         image = cv2.circle(image, (int(co_tay_phai[0]), int(
#             co_tay_phai[1])), radius=5, color=(0, 0, 255), thickness=-1)  # vaitrai

#         ab = math.sqrt((canhtay_phai[0]-canhtay_trai[0])
#                        ** 2 + (canhtay_phai[1]-canhtay_trai[1])**2)
#         ab = math.sqrt((co_tay_phai[0]-co_tay_trai[0])
#                        ** 2 + (co_tay_phai[1]-co_tay_trai[1])**2)

#         if (vai_phai_y[1] > 0 and co_tay_phai[1] > 0):
#             if (co_tay_phai[1] <= vai_phai_y[1]):
#                 print("warning")
#         if (vai_trai_y[1] > 0 and co_tay_trai[1] > 0):
#             if (co_tay_trai[1] <= vai_trai_y[1]):
#                 print("warning")
#         khoang_cach_canh_tay = math.sqrt(
#             (canhtay_phai[0]-canhtay_trai[0])**2 + (canhtay_phai[1]-canhtay_trai[1])**2)
#         khoang_cach_co_tay = math.sqrt(
#             (co_tay_phai[0]-co_tay_trai[0])**2 + (co_tay_phai[1]-co_tay_trai[1])**2)
#         if khoang_cach_canh_tay == 0 and khoang_cach_co_tay == 0:
#             continue
#         if khoang_cach_canh_tay <= 70 and khoang_cach_co_tay <= 35:
#             print("warning")
#         kp += 1

# cv2.imshow('test', image)
# cv2.waitKey(0)

# test video

cap = cv2.VideoCapture("rob_all.mp4")
count = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
vid_out = cv2.VideoWriter('out_put.avi',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          24, size)
while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        break
    results = model.track(frame, stream=True)
    for r in results:
        img = r.orig_img
        keypoints = r.keypoints.data.tolist()
        frame_with_box = r.plot()
        boxes = r.boxes.data.tolist()
        kp = 0
        for box in boxes:
            vai_phai_y = (keypoints[kp][6])
            canhtay_phai = keypoints[kp][8]
            co_tay_phai = (keypoints[kp][10])

            vai_trai_y = (keypoints[kp][5])
            canhtay_trai = keypoints[kp][7]
            co_tay_trai = (keypoints[kp][9])
            if (vai_phai_y[1] > 0 and co_tay_phai[1] > 0):
                if (co_tay_phai[1] <= vai_phai_y[1]):
                    frame = cv2.putText(frame, 'Warning', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                    break
            if (vai_trai_y[1] > 0 and co_tay_trai[1] > 0):
                if (co_tay_trai[1] <= vai_trai_y[1]):
                    frame = cv2.putText(frame, 'Warning', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                    break
            khoang_cach_canh_tay = math.sqrt(
                (canhtay_phai[0]-canhtay_trai[0])**2 + (canhtay_phai[1]-canhtay_trai[1])**2)
            khoang_cach_co_tay = math.sqrt(
                (co_tay_phai[0]-co_tay_trai[0])**2 + (co_tay_phai[1]-co_tay_trai[1])**2)
            if khoang_cach_canh_tay == 0 and khoang_cach_co_tay == 0:
                continue
            if khoang_cach_canh_tay <= 70 and khoang_cach_co_tay <= 35:
                frame = cv2.putText(frame, 'Warning', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3, cv2.LINE_AA)
                break
            kp += 1
        count += 1
        vid_out.write(frame)
        cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from transforms.eye_crop import RandomCrop, Centercrop


class Crop:
    def __init__(self, size=224, random_scale=(0.2, 1.0), center_scale=(0.6, 1)):
        self.size = size
        self.random_scale = random_scale
        self.center_scale = center_scale

    def find_eye(self, image):
        face_landmarks = face_recognition.face_landmarks(np.array(image))
        if face_landmarks == []:
            return None
        else:
            # 获取左眼和右眼的坐标
            left_eye = np.array(face_landmarks[0]['left_eye'])
            right_eye = np.array(face_landmarks[0]['right_eye'])

            # 获取左眼和右眼的矩形框
            left_eye_box = (min(left_eye[:, 0]) - 10, min(left_eye[:, 1] - 10),
                            max(left_eye[:, 0]) + 10, max(left_eye[:, 1]) + 10)

            # 计算右眼框的坐标
            right_eye_box = (min(right_eye[:, 0]) - 10, min(right_eye[:, 1] - 10),
                             max(right_eye[:, 0]) + 10, max(right_eye[:, 1]) + 10)

            eye_region = left_eye_box + right_eye_box
            return eye_region

    def rectangle(self, image, eye_region):
        img = image
        a = ImageDraw.ImageDraw(img)
        a.rectangle(((eye_region[0], eye_region[1]), (eye_region[2], eye_region[3])), outline='green', width=2)
        a.rectangle(((eye_region[4], eye_region[5]), (eye_region[6], eye_region[7])), outline='green', width=2)
        return img

    def __call__(self, image):
        eye_region = self.find_eye(image)
        if eye_region is not None:
            transform = RandomCrop(size=self.size, scale=self.random_scale,
                                   eye_region=eye_region)
        else:
            transform = Centercrop(self.size, scale=self.center_scale)
        return transform(image)


if __name__ == '__main__':
    image = Image.open(r"F:\Gaze360\Gaze360-New\Image\train\Face\14124.jpg").convert('RGB')

    trans = Crop()
    transform = trans(image)
    image1 = transform(image)
    # image2 = transform(image)
    image1.show()
    image1.save('eye_region_crop1.png')
    # image2.show()
    # image2.save('eye_region_crop2.png')



# import cv2
# import numpy as np
# from PIL import Image, ImageDraw
# from transforms.eye_crop import RandomCrop, Centercrop
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import logging
# import mediapipe as mp


# class Crop():
#     def __init__(self, size=224, random_scale=(0.2, 1.0), center_scale=(0.6, 1)):
#         self.size = size
#         self.random_scale = random_scale
#         self.center_scale = center_scale

#     def find_eye(self, image):
#         face_mesh = mp.solutions.face_mesh.FaceMesh()
#         result = face_mesh.process(np.array(image))
        
#         if result.multi_face_landmarks is None:
#             return None
        
#         if result.multi_face_landmarks:
#             for face in result.multi_face_landmarks:
#                 # 获取眼睛特定特征点
#                 right_eye_landmark_ids = [362, 385, 387, 263, 373, 380]
#                 left_eye_landmark_ids = [33, 160, 158, 133, 153, 144]
#                 lx = []; ly =[]
#                 rx = []; ry =[]
#                 for id, landmark in enumerate(face.landmark):
#                     if id in right_eye_landmark_ids:
#                         x = int(landmark.x * self.size)
#                         y = int(landmark.y * self.size)
#                         lx.append(x); ly.append(y)
#                     if id in left_eye_landmark_ids:
#                         x = int(landmark.x * self.size)
#                         y = int(landmark.y * self.size)
#                         rx.append(x); ry.append(y)

#                 left_eye = (min(lx) - 10, min(ly) - 10, max(lx) + 10, max(ly) + 10)
#                 right_eye = (min(rx) - 10, min(ry) - 10, max(rx) + 10, max(ry) + 10)

#                 eye_region = left_eye + right_eye
#                 return eye_region

#     # def rectangle(self, image, left_eye, right_eye):
#     #     a = ImageDraw.Draw(image)
#     #     a.rectangle(((left_eye[0], left_eye[1]), (left_eye[2], left_eye[3])), outline="red")
#     #     a.rectangle(((right_eye[0], right_eye[1]), (right_eye[2], right_eye[3])), outline="red")

#     def __call__(self, image):
#         eye_region = self.find_eye(image)
#         if eye_region is not None:
#             transform = RandomCrop(size=self.size, scale=self.random_scale,
#                                    eye_region=eye_region)
#         else:
#             transform = Centercrop(self.size, scale=self.center_scale)
#         return transform(image)


# if __name__ == '__main__':
#     image = Image.open(r'F:\MPIIFaceGaze\MPIIFaceGaze-new\Image\p14\face\1233.jpg').convert('RGB')

#     trans = Crop()
#     transform = trans(image)
#     image1 = transform(image)
#     image1.show()


import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

def extract_250(path):
    empty = []
    cap  = cv2.VideoCapture(path)
    k=0
    while cap.isOpened():
        ret, frame = cap.read()
        img_ = cv2.resize(frame, (480, 270))
        k+=1
        if k==500:
            empty.append(img_)
            cap.release()
    return empty

def cam_loc_checker(data_path):
  font =  cv2.FONT_HERSHEY_PLAIN

  working_list = pd.DataFrame({'working':[i[:-3] for i in os.listdir(data_path)]}).drop_duplicates()['working'].tolist()
  font =  cv2.FONT_HERSHEY_PLAIN

  for i in working_list[1:3]:
    # C1
    C1_path = data_path+'{}_C1/{}_C1.mp4'.format(i,i)
    img_C1 = extract_250(C1_path)
    C1_img=(img_C1[0])
    # # 이미지에 글자 합성하기
    C1_img = cv2.putText(C1_img, "C1", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C1_img)

    #C2
    C2_path = data_path+'{}_C2/{}_C2.mp4'.format(i,i)
    img_C2 = extract_250(C2_path)
    C2_img=(img_C2[0])
    # # 이미지에 글자 합성하기
    C2_img = cv2.putText(C2_img, "C2", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C2_img)
    
    #C3
    C3_path = data_path+'{}_C3/{}_C3.mp4'.format(i,i)
    img_C3 = extract_250(C3_path)
    C3_img=(img_C3[0])
    # # 이미지에 글자 합성하기
    C3_img = cv2.putText(C3_img, "C3", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C3_img)

    #C4
    C4_path = data_path+'{}_C4/{}_C4.mp4'.format(i,i)
    img_C4 = extract_250(C4_path)
    C4_img=(img_C4[0])
    # # 이미지에 글자 합성하기
    C4_img = cv2.putText(C4_img, "C4", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C4_img)

    #C5
    C5_path = data_path+'{}_C5/{}_C5.mp4'.format(i,i)
    img_C5 = extract_250(C5_path)
    C5_img=(img_C5[0])
    # # 이미지에 글자 합성하기
    C5_img = cv2.putText(C5_img, "C5", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C5_img)
    
    #C6
    C6_path = data_path+'{}_C6/{}_C6.mp4'.format(i,i)
    img_C6 = extract_250(C6_path)
    C6_img=(img_C6[0])
    # # 이미지에 글자 합성하기
    C6_img = cv2.putText(C3_img, "C6", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C3_img)

    #C7
    C7_path = data_path+'{}_C7/{}_C7.mp4'.format(i,i)
    img_C7 = extract_250(C7_path)
    C7_img=(img_C7[0])
    # # 이미지에 글자 합성하기
    C7_img = cv2.putText(C7_img, "C7", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C7_img)

    #C8
    C8_path = data_path+'{}_C8/{}_C8.mp4'.format(i,i)
    img_C8 = extract_250(C8_path)
    C8_img=(img_C8[0])
    # # 이미지에 글자 합성하기
    C8_img = cv2.putText(C8_img, "C8", (430, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(C8_img)
    
    #middle
    black = np.zeros((270, 480,3), dtype=np.uint8)
    black = cv2.putText(black, "{}".format(i), (120, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    #show(black)
    
    ### total
    line_1 = cv2.hconcat([C1_img, C2_img, C3_img])
    line_2 = cv2.hconcat([C4_img, black, C5_img])
    line_3 = cv2.hconcat([C6_img, C7_img, C8_img])
    total = cv2.vconcat([line_1, line_2, line_3])

    cv2.imwrite('results/cam_loc_{}.jpg'.format(i), total)

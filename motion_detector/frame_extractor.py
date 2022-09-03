import cv2
from tqdm import tqdm
import os
import pandas as pd
import sys
# from FINAL_IMG_PREP import *
from video_info import *

folder_path = '/Users/monica_air/Documents/CCTV/1. 해외환경(1500개)/1. 배회(325개)'

list_ = os.listdir(folder_path)
file_list=[]
for i in list_:
    if '.mp4'in i:
        file_list.append(i)
        
start_list_v2= []
end_list_v2=[]
file_name_v2 = []

for i in tqdm (file_list):
    try:
        path = '/Users/monica_air/Documents/CCTV/1. 해외환경(1500개)/1. 배회(325개)/'+i
            
        cap, width, height, fps, fourcc, filename, out = vid_info(path, 'XVID','0903_test')
        # print(fps)
        if not cap.isOpened():
            print('Video open failed!')
            sys.exit()

        # 배경 영상 등록
        ret, back = cap.read()

        if not ret:
            print('Background image registration failed!')
            sys.exit()
            
        # 연산 속도를 높이기 위해 그레이스케일 영상으로 변환
        back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러로 노이즈 제거 (모폴로지, 열기, 닫기 연산도 가능)
        back = cv2.GaussianBlur(back, (0, 0), 1.0)
        frame_num = []
        z=0
        # 비디오 매 프레임 처리
        while True:
            ret, frame = cap.read()
            z+=1
            if not ret:
                break
            
            # 현재 프레임 영상 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 노이즈 제거
            gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
            
            # 차영상 구하기 $ 이진화
            # absdiff는 차 영상에 절대값
            diff = cv2.absdiff(gray, back)
            # gray_mean() or 밝기정도 지정 가능
            ####!!!!!back.mean()으로 하면 smoke까지 가능?
            _, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
            
            # 레이브링을 이용하여 바운딩 박스 표시
            cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)
            
            #contour
        #     dilated = cv2.dilate(diff, None, iteration=3)

            # contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            
            for i in range(1, cnt):
                x, y, w, h, s = stats[i]
                
                if s < 100: # 작을 수록 더 많이 추출
                    continue
                
                # cv2.drawContours(frame, contours, -1, (0, 0, 255),2)
                if x+y+w+h >=500:
                    pass
                    # cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 2)
                    # cv2.putText(frame, "Frame:{}".format(z), (10,20),cv2.FONT_HERSHEY_SIMPLEX,
                    #         1, (0,0,255),3)
                    frame_num.append(z)
        start = str(frame_num[0])
        end = str(frame_num[-1])
        
        file_name_v2.append(file_list[i])
        start_list_v2.append(start)
        end_list_v2.append(end)
    except:
        pass
    

df = pd.DataFrame({'file_name':file_name_v2, 'start':start_list_v2, 'end':end_list_v2})
df.to_csv('1. 해외환경\(1500개\).csv')
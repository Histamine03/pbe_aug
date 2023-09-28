import os
import cv2
import random
import sys
import numpy as np
import yaml
# Add the subdirectory to the Python path

# Import the main_script module from the subdirectory
from scripts import inference
from tqdm import tqdm
import argparse

#takes bbox coords => returns yolo style bbox coords 
#yolo 포맷으로 bbox 좌표 변환
def bbox2yolo(xmin, ymin, xmax, ymax, img_shape):
    b_center_x = (xmin + xmax) / 2 
    b_center_y = (ymin + ymax) / 2
    b_width    = (xmax - xmin)
    b_height   = (ymax - ymin)
    image_h, image_w, image_c = img_shape
    b_center_x /= image_w 
    b_center_y /= image_h 
    b_width    /= image_w 
    b_height   /= image_h 
    return b_center_x, b_center_y, b_width, b_height


#randomly crops img by 1/rows*cols size => returns cropped img, cropped coordinates, random selceted # of row
def crop_rand_region(img, rows, cols):
    print('crop rand region called')
    random.seed()
    original_image = img
    height, width, _ = original_image.shape
    cropped_width = width // cols
    cropped_height = height // rows
    row = random.randrange(rows)
    col = random.randrange(cols)

    left = col * cropped_width
    upper = row * cropped_height
    right = left + cropped_width
    lower = upper + cropped_height
    
    cropped_image = original_image[upper:lower, left:right]
    return cropped_image, [upper,lower, left, right], row

def pbe_aug(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lst = config['mp4_list']
    rt = config['rt']
    save_dir = config['save_dir']
    mask = config['mask_dir']
    gan_img = config['gen_img']
    model_path_ = config['model_path']

    # test 데이터셋 동영상 파일명 list

    #pbe

    #SD Model initialization
    model = inference.init_model_ret(model_path = model_path_)
    print('model initialized!')

    #root directory of dataset

    count = 0
    #데이터셋 구조 따라 iterate
    for subdir in tqdm(os.listdir(rt)):
        if subdir == 'pbe_aug_res' : 
            continue
        for subsubdir in os.listdir(os.path.join(rt,subdir)):
            if not subsubdir.startswith('.'):
                for file in os.listdir(os.path.join(rt,subdir,subsubdir)):
                    if not file.startswith('.') and file not in lst and '.mp4' in file:
                        print(os.path.join(rt,subdir,subsubdir,file))
                        fname = file.strip('.mp4')
                        
                        #영상 파일 오픈
                        cap = cv2.VideoCapture(os.path.join(rt,subdir,subsubdir,file))
                        frame_count =0 #프레임 수 

                        #영상 내 프레임들 모두 iterate 하여 처리
                        while True:
                            ret_val, frame = cap.read() 
                            #4초에 프레임 한장씩 받아옴
                            if frame_count%120 != 0:
                                frame_count+=1
                                continue
                            if not ret_val:
                                break
                            #1/64 사이즈로 random crop
                            cropped_img, coords, row = crop_rand_region(frame,8,8)
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                            #sd 모델 이nsfw outout을 return 할 경우 black image 를 반환함. 이를 대비하여 black image 일 경우 재추론
                            while True:
                                ret_img = inference.main(model, mask ,cropped_img, gan_img)
                                ret_img = cv2.cvtColor(np.asarray(ret_img), cv2.COLOR_RGB2BGR)

                                mean_val = ret_img.mean()
                                if mean_val<10:
                                    print('black img found.')
                                else:
                                    break  

                            #crop된 이미지 별도 저장
                            upper,lower, left, right = coords
                            #원본 frame에 paint by example 결과물 재합성
                            frame[upper:lower, left:right] = ret_img
                            frame_count+=1
                            
                            #최종 결과물 저장
                            cv2.imwrite(f'{save_dir}/{fname}_pbe_{frame_count}.jpg', frame)

                            #bbox 좌표 yolo style 로 변환 후 txt 파일로 저장
                            mask_image = cv2.imread(mask , cv2.IMREAD_GRAYSCALE)
                            mask_image = cv2.resize(mask_image, (512, 512))
                            ret_h,ret_w,c = ret_img.shape
                            mask_image = cv2.resize(mask_image, (ret_w,ret_h))  

                            white_pixels_mask = (mask_image == 255)  
                            white_pixel_indices = np.where(white_pixels_mask)
                            #mask의 흰색 영역 좌표 계산
                            min_x = np.min(white_pixel_indices[1])
                            min_y = np.min(white_pixel_indices[0])
                            max_x = np.max(white_pixel_indices[1])
                            max_y = np.max(white_pixel_indices[0])

                            #bbox 좌표 txt 파일로 저장
                            bbox = min_x+left, min_y+upper, max_x+left, max_y+upper
                            yolobbox = bbox2yolo(min_x+left, min_y+upper, max_x+left, max_y+upper, frame.shape) #yolo style로 bbox coords 변환
                            with open(f'{save_dir}/{fname}_pbe_{frame_count}.txt', 'w') as txt_result:
                                txt_result.write(f"0 {yolobbox[0]} {yolobbox[1]} {yolobbox[2]} {yolobbox[3]}")

                            print(f'imgs saved for {file} @ frame#{frame_count}. coords: {coords}, row: {row}')

                        #video source file read 종료
                        if cap.isOpened():	
                            cap.release()	

def config():

    parser = argparse.ArgumentParser(description = "yaml file path")
    parser.add_argument('--yaml_path', type = str, default = 'config.yaml')
    args = parser.parse_args()

    return args.yaml_path

if __name__ == "__main__":
    yaml_path = config()
    pbe_aug(yaml_path)

breakpoint()

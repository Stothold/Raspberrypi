import ctypes
import time
import onnxruntime
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2 #preview 필요없는거 확인해 삭제
import cv2

# 공유 라이브러리 로드 (led_control.so, blackjack_strategy.so)
blackjack_lib = ctypes.CDLL('./libblackjack2.so')

blackjack_lib.calculate_points.argtypes = [ctypes.c_char_p, ctypes.c_int]
blackjack_lib.calculate_points.restype = ctypes.c_int
blackjack_lib.get_optimal_move.argtypes = [ctypes.c_char_p, ctypes.c_int]
blackjack_lib.get_optimal_move.restype = ctypes.c_bool

LED=4
GPIO.setmode(GPIO.BCM) 
GPIO.setup(LED, GPIO.OUT, initial=GPIO.LOW) #LED gpio 기초 설정

camera = Picamera2() 
camera.configure(camera.create_preview_configuration()) #카메라 기초 설정


def non_max_suppression(prediction, conf_thres, iou_thres):
    mask = prediction[:,4]>conf_thres
    prediction = prediction[mask]
    
    if len(prediction) ==0:
        return np.array([]),np.array([])
        
    boxes = prediction[:,:4]
    scores = prediction[:,4] #*np.max(prediction[:,5:],axis=1)
    class_ids = np.argmax(prediction[:,5:],axis = 1)
    
    areas = (boxes[:,2] - boxes[:,0]) *(boxes[:,3] - boxes[:,1])
    
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size ==1:
            break

        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.maximum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.maximum(boxes[i,3], boxes[order[1:],3])
        
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w * h
        ovr = inter/(areas[i] + areas[order[1:]] -inter)
        
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds +1]
        
    return scores[keep], class_ids[keep].astype(int)

# 카드 클래스 이름
classNames = ['T', 'T', 'T', 'T',
              '2', '2', '2', '2',
              '3', '3', '3', '3',
              '4', '4', '4', '4',
              '5', '5', '5', '5',
              '6', '6', '6', '6',
              '7', '7', '7', '7',
              '8', '8', '8', '8',
              '9', '9', '9', '9',
              'A', 'A', 'A', 'A',
              'J', 'J', 'J', 'J',
              'K', 'K', 'K', 'K',
              'Q', 'Q', 'Q', 'Q']

# ONNX 모델 로드
ort_session = onnxruntime.InferenceSession('blackjack7.onnx')

# 코드 작동 시간 측정
camera.start()
start = time.time()

while True:
    im = camera.capture_array()
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # 이미지 크기 조정
    img = img.astype(np.float32) / 255.0

    # 차원 변환 (channels, height, width) 및 배치 차원 추가
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    # ONNX 모델 추론
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    before_nms = ort_outs[0][0]
    conf, results = non_max_suppression(before_nms, conf_thres=0.5, iou_thres=0.1)
    
    results = list(set(results))
    
    hand = [classNames[cls] for cls in results]
    
    # C 함수 호출 (ctypes 사용, 이 부분은 변경 없음)
    hand_array = (ctypes.c_char * len(hand))(*[c.encode() for c in hand])
    hand_size = len(hand)
    calc = blackjack_lib.get_optimal_move(hand_array, hand_size)
    if calc == 1: #stand일때 LED 켜기
        GPIO.output(LED, True)
        
    else: #hit일때 LED 끄기
        GPIO.output(LED, False)
    
    print(time.time()-start) #프로그램 작동 시간 측정. while 루프 안에 위치해 있어서 측정이 필요 이상으로 될 것 같으나 뾰족한 해결책 없어서 일단 여기 위치시킴.



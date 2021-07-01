import numpy as np
from PIL import Image
from yolo_net.Class_Yolo import YOLO
import  cv2
if __name__ == '__main__':
    yolo = YOLO()
    cap = cv2.VideoCapture(0)
    w, h = 640, 480
    while True:
        kk = cv2.waitKey(1)
        # 按下英文下的 q 键退出
        # press 'q' to exit
        if kk == ord('q'):
            break
        r, frame = cap.read()q
        frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
        #cv2.imshow('1',frame)
        #cv2.waitKey(100)
        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = yolo.detecter(frame)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        #cvtColor()颜色空间转换函数
    cap.release()
    cv2.destroyAllWindows()  # 关闭窗口
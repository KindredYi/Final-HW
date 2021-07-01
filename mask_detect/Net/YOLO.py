from PIL import ImageDraw,ImageFont
from Net.YOLO_Net import YoloBlock
from Utils.utils import DecodeBox,letterbox,NMS,remove_gray
import numpy as np
import torch
import os

class YOLO(object):
    result=[]
    keyval=[0,0]
    defaults={
        "model_path":r"D:\PycharmProjects\Mask_Detection\model\time=20210125_loss=2.016.pth",
        "anchor_path":r'D:\PycharmProjects\Mask_Detection\utils\kmeans_anchor.txt',
        "class_path":r"D:\PycharmProjects\Mask_Detection\utils\classes.txt",
        "image_shape":(416,416,3),
        "confidence":0.90
    }
    @classmethod
    def get_defaults(cls,n):
        if n in cls.defaults:
            return cls.defaults[n]
        else:
            return "key error"

    def __init__(self,**kwargs):
        self.__dict__.update(self.defaults)
        self.class_name=self.get_class()
        self.anchor=self.get_anchors()
        self.generate()

    def get_class(self):
        class_path = os.path.expanduser(self.class_path)
        with open(class_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchor_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])
    def generate(self):
        self.net=YoloBlock(3,2).eval()
        device=torch.device("cpu")
        model_dict=torch.load(self.model_path,map_location=device)
        self.net.load_state_dict(model_dict)
        print("从“{}”加载模型".format(self.model_path))

        self.feat_decoder=[]
        self.anchor_arr=[[3,4,5],[0,1,2]]
        for i in range(2):
            decoder=DecodeBox(np.reshape(self.anchor,[-1,2])[self.anchor_arr[i]],2,(416,416))
            # print('创建第{}个解码器'.format(i))
            self.feat_decoder.append(decoder)

    def detecter(self,image):
        self.result.clear()
        image_shape=np.array(np.shape(image)[0:2])
        new_img=letterbox(image,(416,416))
        crop_img=np.array(new_img)
        img=np.array(crop_img,dtype=np.float32)/255.0
        img=np.transpose(img,(2,0,1))

        imgs=[img]


        with torch.no_grad():
            in_img=torch.from_numpy(np.asarray(imgs))
            out_feat=self.net(in_img)
            out_list=[]
            for i in range(2):
                print('*****'*10)
                print('加载第{}个解码器'.format(i))
                decoder_out=self.feat_decoder[i](out_feat[i])
                out_list.append(decoder_out)
            output=torch.cat(out_list,dim=1)
            print('*****'*10)
            print('解码器输出融合:{}'.format(output.shape))
            final_detection=NMS(output)

            try:
                batch_detection=final_detection[0].cpu().numpy()
            except:
                return image,self.result
            end_score=batch_detection[:,4]*batch_detection[:,5]
            end_index=end_score>self.confidence

            end_label=np.array(batch_detection[end_index,-1],np.int32)
            end_boxes=np.array(batch_detection[end_index,:4])
            end_xmin = np.expand_dims(end_boxes[:, 0], -1)
            end_ymin = np.expand_dims(end_boxes[:, 1], -1)
            end_xmax = np.expand_dims(end_boxes[:, 2], -1)
            end_ymax = np.expand_dims(end_boxes[:, 3], -1)
            end_boxes=remove_gray(end_ymin,end_xmin,end_ymax,end_xmax,np.array([416,416]),image_shape)
            font = ImageFont.truetype(font='../model/msyh.ttc',
                                      size=np.floor(3e-2 * np.shape(image)[1] +20).astype('int32'))
            line_width = max((np.shape(image)[0] + np.shape(image)[1]) // 416, 1)
            for i ,c in enumerate(end_label):
                predict_class=self.class_name[c]
                score=end_score[i]
                print('score:{}'.format(score))
                top,left,bottem,right=end_boxes[i]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottem = min(np.shape(image)[0], np.floor(bottem + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

                if predict_class=="with_mask":
                    rc="有口罩"
                    label="有口罩：{:.3f}".format(score)
                    color=(0,255,0)
                else:
                    rc="无口罩"
                    label = "无口罩：{:.3f}".format(score)
                    color=(255,0,0)



                label_size=font.getsize(text=label)
                label_w=label_size[0]
                label_h=label_size[1]
                # print(font.getsize(text=label))
                draw=ImageDraw.Draw(image)
                label = label.encode('utf-8')
                label_size=draw.textsize(label)


                if top-label_size[1]>5:
                            #   标签绘制在框上方
                    text_start=[left,top-label_size[1]-10]
                else:       #   标签绘制在框下方
                    text_start=[left,top-4]
                text_x,text_y=text_start[0],text_start[1]
                for i in range(line_width):
                    draw.rectangle((left+i,top+i,right-i,bottem-i),outline=color,width=2)
                draw.rectangle((text_x,text_y,text_x+label_w,text_y+label_h),fill=color)
                draw.text(text_start,str(label,"UTF-8"),fill=(0,0,0),font=font)
                del draw

                self. keyval[0],self.keyval[1]=rc,score
                self.result.append(tuple(self.keyval))
            return image,self.result
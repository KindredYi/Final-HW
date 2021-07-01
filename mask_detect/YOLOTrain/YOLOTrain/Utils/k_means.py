
from PIL import Image
import numpy as np


def iou(box,cluster):
    x=np.minimum(cluster[:,0],box[0])
    y=np.minimum(cluster[:,1],box[1])
    i=x*y
    area1=box[0]*box[1]
    arer2=cluster[:,0]*cluster[:,1]
    IoU=i/(area1+arer2-i)
    return IoU
def kmeans(box,k):
    box_num=box.shape[0]
    distance=np.empty((box_num,k))
    clu=np.zeros((box_num,))
    np.random.seed()
    cluster=box[np.random.choice(box_num,k,replace=False)]
    while True:
        for i in range(box_num):
            distance[i]=1-iou(box[i],cluster)
        nearst=np.argmin(distance,axis=1)
        if (clu==nearst).all():
            break
        for j in range(k):
            cluster[j]=np.median(box[nearst==j],axis=0)
        clu=nearst
    return cluster
def read_txt(path):
    data=[]
    f=open(path)
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        imgpath=line[:52]
        img=Image.open(imgpath)
        w,h=int(img.size[0]),int(img.size[1])
        boxes=line[53:]
        boxes_format=boxes.split()
        for i in boxes_format:
            box=i.split(',')
            print(i)
            xmin=np.float64(int(box[0])/w)
            ymin=np.float64(int(box[1])/h)
            xmax=np.float64(int(box[2])/w)
            ymax=np.float64(int(box[3])/h)
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)



if __name__=='__main__':
    txtpath=r'../logs/trainlist_20000.txt'
    size=416
    anc_num=6
    hwdata=read_txt(txtpath)
    out_anchor=kmeans(hwdata,anc_num)
    out_anchor=out_anchor[np.argsort(out_anchor[:,0])]
    hwdata=out_anchor*416
    f=open('../logs/kmeans_anchor.txt', 'w')
    num=np.shape(hwdata)[0]
    for i in range(num):
        if i==0:
            x_y="%d,%d"%(hwdata[i][0],hwdata[i][1])
        else:
            x_y=", %d,%d"%(hwdata[i][0],hwdata[i][1])
        f.write(x_y)
    f.close()
    print('已生成先验框')
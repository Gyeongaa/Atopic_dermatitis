import numpy as np
import cv2
import torch
import time
import os
import glob
import time
from datetime import datetime
DEVICE = 'cpu'
mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])
model = torch.load('./model.pth', map_location = DEVICE)
# model.eval()
mask_list = glob.glob('/tmp/data/*/crop/lesion_area/*.png')
test_list = glob.glob('/tmp/data/*/crop/*.jpg')
total_dsc=0
timestamp = time.time()
kst_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
f=open('/tmp/data/atopic_lesion_test_log.txt','w')
k=0
f.write('python3 main.py\n')
f.write(kst_time)
f.write('\nTime\tFilename\tTP\tFP\tFN\tDSC\n')
for i, filename in enumerate(test_list):
    image = cv2.imread(filename)
    gt_mask = cv2.imread(mask_list[i],0)
    gt_mask = gt_mask.astype(np.uint8)
    gt_mask[gt_mask>0] = 1
    gt=gt_mask.flatten()  
    image_input = cv2.resize(image, (256,256))
    image_input_normalize = (image_input / 255 - mean) / std
    transpose_image = image_input_normalize.transpose(2, 0, 1).astype('float32')
    x_tensor = torch.from_numpy(transpose_image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = pr_mask.astype(np.uint8)
    pr_mask = cv2.resize(pr_mask,image.shape[1::-1])
    pr_mask[pr_mask>0]=255
    gt_mask = cv2.imread(mask_list[i],0)

    pr_mask[pr_mask>0] = 1
    gt_mask[gt_mask>0] = 1
    gt=gt_mask.flatten()        
    # print(gt_mask.shape, pr_mask.shape)
    
    pred=pr_mask.flatten()
    intersection = np.sum(gt*pred)
    tp = intersection
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    dice = (2*tp)/(2*tp + fp + fn)
    total_dsc = total_dsc + dice
    file_name_split = filename.split('/')
    # print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'\t'+os.path.join(file_name_split[3],file_name_split[4],file_name_split[5],file_name_split[6])+'\tIOU: %.4f\tIntersection: %d\tUnion: %d\n'%(iou,intersection,union))
    f.write(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'\t'+os.path.basename(filename)[:-4]+'\t%d\t%d\t%d\t%.4f\n'%(tp,fp,fn, dice))
    k=k+1

# print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'\tAverage IOU: ',total_iou/k)
f.write(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'\tAverage DSC: '+str(total_dsc/k)+'\n')
# timestamp = time.time()
# kst_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
# f.write(str(timestamp)+'\t'+kst_time)
f.close()
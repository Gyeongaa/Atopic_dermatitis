# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from scipy.special import softmax
import glob
from PIL import Image
import json
from sklearn.preprocessing import label_binarize
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class_names = ['Normal', 'Almost Clear', 'Mild', 'Moderate', 'Severe']
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode = 'Training', transform=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_path = glob.glob(root_dir+'/*/crop/*.jpg')
        self.transform = transform
        self.mode = mode
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.mode == 'Test':
            return img, self.img_path[idx]
        else:
            dir_name = os.path.dirname(self.img_path[idx])
            file_name = os.path.basename(self.img_path[idx])
            json_full_path = os.path.join(os.path.join(dir_name[:-4], 'metadata'), file_name[:-4] + '.json')
            if os.path.exists(os.path.join(dir_name,'lesion_area')):
                data = json.load(open(json_full_path))
            else:
                data = json.load(open(json_full_path, encoding='cp949'))
            if 'iga_grade' in data['annotations'][0]['clinical_info']:
                grade = class_names.index(data['annotations'][0]['clinical_info']['iga_grade'])
            else:
                grade = 0
            return img, grade, file_name[:-4]

test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

f=open('/tmp/data/atopic_severity_classification_auc_test_log.txt','w')
f.write('python3 main.py\n')
timestamp = time.time()
kst_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
f.write(kst_time)
f.write('\nTime\tClassID\tTP\tTN\tFP\tFN\tP\tN\tThreshold\tTNR\tFPR\tFPR\n')
class0_total = 0
class0_correct = 0
class1_total = 0
class1_correct = 0
class2_total = 0
class2_correct = 0
class3_total = 0
class3_correct = 0
class4_total = 0
class4_correct = 0
total = 0
correct = 0
if __name__=='__main__':
    device = 'cpu'
    # data_dir = r'E:\2021피부과과제\피부과과제_제출\아토피피부염_병변_분할_정확도\평가용 데이터셋'
    data_dir = '/tmp/data'
    model = torch.load('model.pth', map_location=device)
    model.eval()
    test_dataset = CustomDataset(data_dir, transform = test_transforms)

    dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
    device = torch.device(device)
    y=[]
    y_score = np.zeros([856,5])
    k=0
    folder_list=[]
    with torch.no_grad():
        for inputs, labels, folder_name in dataloaders:
            folder_list.append(folder_name[0])
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            output = outputs.cpu().numpy()
            output_softmax = softmax(output)
            max_value = np.max(output_softmax)
            max_index = np.argmax(output_softmax)
            if (labels.numpy()[0] == 0):
                class0_total = class0_total + 1
                if (labels.numpy()[0] == max_index):
                    class0_correct = class0_correct + 1
                    correct = correct + 1
            if (labels.numpy()[0] == 1):
                class1_total = class1_total + 1
                if (labels.numpy()[0] == max_index):
                    class1_correct = class1_correct + 1
                    correct = correct + 1
            if (labels.numpy()[0] == 2):
                class2_total = class2_total + 1
                if (labels.numpy()[0] == max_index):
                    class2_correct = class2_correct + 1
                    correct = correct + 1
            if (labels.numpy()[0] == 3):
                class3_total = class3_total + 1
                if (labels.numpy()[0] == max_index):
                    class3_correct = class3_correct + 1
                    correct = correct + 1
            if (labels.numpy()[0] == 4):
                class4_total = class4_total + 1
                if (labels.numpy()[0] == max_index):
                    class4_correct = class4_correct + 1
                    correct = correct + 1
            total = total + 1
            y.append(labels.numpy()[0])
            y_score[k,:] = np.array([output_softmax[0][0], output_softmax[0][1],output_softmax[0][2],output_softmax[0][3],output_softmax[0][4]])
            k=k+1

    y_test=label_binarize(y, classes=[0,1,2,3,4])
    n_classes = y_test.shape[1]
    
    auc = [0 for i in range(5)]
    for i in range(5):
        y_score_class = y_score[:,i]
        score_ind = np.argsort(y_score_class)
        y_sort = np.array(y)[score_ind]
        y_score_class_sort = np.array(y_score_class)[score_ind]
        folder_class_sort = np.array(folder_list)[score_ind]
        threshold_list = np.unique(y_score_class_sort)
        threshold_list = threshold_list[::-1]
        tp=[0 for i in range(len(threshold_list))]
        tn=[0 for i in range(len(threshold_list))]
        fp=[0 for i in range(len(threshold_list))]
        fn=[0 for i in range(len(threshold_list))]
        p=[0 for i in range(len(threshold_list))]
        n=[0 for i in range(len(threshold_list))]
        tnr=[0 for i in range(len(threshold_list))]
        fpr=[0 for i in range(len(threshold_list))]
        tpr=[0 for i in range(len(threshold_list))]
        for j in range(len(threshold_list)):
            thres_over_score = y_score_class_sort[y_score_class_sort>=threshold_list[j]]
            y_over_score = y_sort[y_score_class_sort>=threshold_list[j]]
            thres_under_score = y_score_class_sort[y_score_class_sort<threshold_list[j]]
            y_under_score = y_sort[y_score_class_sort<threshold_list[j]]
            
            tp[j] = np.sum(y_over_score == i)
            tn[j] = np.sum(y_under_score != i)
            fp[j] = np.sum(y_over_score != i)
            fn[j] = np.sum(y_under_score == i)
            
            p[j] = tp[j] + fn[j]
            n[j] = tn[j] + fp[j]
            
            tnr[j] = tn[j] / (fp[j] + tn[j])
            fpr[j] = 1 - tnr[j]
            tpr[j] = tp[j] / (tp[j] + fn[j])
            tmp_auc = 0
            if j>0:
                tmp_auc = ((tpr[j]+tpr[j-1])*(fpr[j]-fpr[j-1]))/2
                auc[i] += tmp_auc
            f.write(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+'\t'+class_names[i]+'\t%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%d\t%d\t%d\n'%(tp[j],tn[j],fp[j],fn[j],p[j],n[j],threshold_list[j],tnr[j],fpr[j],tpr[j]))
            # print(tp[j], tn[j], fp[j], fn[j], threshold_list[j], tnr[j], fpr[j], tpr[j], tmp_auc)
    for i in range(5):
        f.write('AUC of Class '+class_names[i]+': %.4f\n'%auc[i])
    avg_auc = np.mean(auc)
    f.write('macro average AUC: %.4f\n'%avg_auc)
    f.write(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    f.close()
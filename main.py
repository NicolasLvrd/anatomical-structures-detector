import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import nibabel as nib
import numpy as np
import csv
from math import dist
import torchvision
from multiprocessing import Process, Array
import multiprocessing
from PIL import Image, ImageDraw
from math import floor
from multiprocessing import Pool
import time


#fold = int(sys.argv[1]) # fold number
fold = 0

transform = torchvision.transforms.Normalize(127.5, 127.5)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5) # 10 x (33-6+1) x 28
        self.pool = nn.MaxPool2d(2) # 10 x 7 x 7
        self.conv2 = nn.Conv2d(6, 16, 5) # 20 x (7-6+1) x 2
        self.fc1 = nn.Linear(400, 150)
        self.fc2 = nn.Linear(150, 46)
        self.fc3 = nn.Linear(150, 50)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

organs = {
    1247 : "Trachea",
    1302 : "Right Lung",
    1326 : "Left Lung",
    170 : "Pancreas",
    187 : "Gallbladder",
    237 : "Urinary Bladder",
    2473 : "Sternum",
    29193 : "First Lumbar Vertebra",
    29662 : "Right Kidney",
    29663 : "Left Kidney",
    30324 : "Right Adrenal Gland",
    30325 : "Left Adrenal Gland",
    32248 : "Right Psoas Major",
    32249 : "Left Psoas Major",
    40357 : "Right rectus abdominis",
    40358 : "Left rectus abdominis",
    480 : "Aorta",
    58 : "Liver",
    7578 : "Thyroid Gland",
    86 : "Spleen",
    0 : "Background",
    1 : "Body Envelope",
    2 : "Thorax-Abdomen"
}

organs_code = {1247: 0, 1302: 1, 1326: 2, 170: 3, 187: 4, 237: 5, 2473: 6, 29193: 7, 29662: 8, 29663: 9, 30324: 10, 30325: 11, 32248: 12, 32249: 13, 40357: 14, 40358: 15, 480: 16, 58: 17, 7578: 18, 86: 19, 0: 20, 1: 21, 2: 22}

patients_code = {
    10000100: 0,
    10000104: 1,
    10000105: 2,
    10000106: 3,
    10000108: 4,
    10000109: 5,
    10000110: 6,
    10000111: 7,
    10000112: 8,
    10000113: 9,
    10000127: 10,
    10000128: 11,
    10000129: 12,
    10000130: 13,
    10000131: 14,
    10000132: 15,
    10000133: 16,
    10000134: 17,
    10000135: 18,
    10000136: 19
}

fold_to_patients = {
    0: [15, 16, 17, 18, 19],
    1: [10, 11, 12, 13, 14],
    2: [5, 6, 7, 8, 9],
    3: [0, 1, 2, 3, 4]
}

group_code = {i: i%5 for i in fold_to_patients[fold]}

def translate(value):
    leftMax = 2976
    leftMin = -1024
    rightMin = 0
    rightMax = 255
    
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan), 5)

translator = np.vectorize(translate)
    
def z_slice(slice, model, z, stride):
    #print(z)
    coor = [[0, 0, 0, 0] for i in range(20)]
    for x in range(floor(1+(512-33)/stride)):
        for y in range(floor(1+(512-33)/stride)):
            imagette = np.array(slice[x*stride:x*stride+33, y*stride:y*stride+33])
            imagette = translator(imagette)
            imagette = torch.from_numpy(imagette)
            imagette = imagette.reshape(1, 1, 33, 33)
            imagette = imagette.type(torch.FloatTensor)
            imagette = transform(imagette)

            pred = model(imagette)
            prob = torch.nn.functional.softmax(pred, dim=1).detach()
            #torch.set_printoptions(precision=2, sci_mode=False)
            sorted, idx = torch.sort(prob, descending=True)
            sorted = sorted.detach().numpy()
            np.set_printoptions(precision=0, suppress=True)
            print(f"prob : {int(sorted[0][0]*100)} ; {int(sorted[0][1]*100)} ; {int(sorted[0][2]*100)}")
              #print(prob.shape)
            guess = torch.argmax(prob, dim=1)
            # print(f"At ({x*33+16};{y*33+16};{z}) : {organs[list(organs_code.keys())[list(organs_code.values()).index(guess.item())]]}")
            
            
            for organ in range(20):
                #if prob[0][organ] > 0.5:
                coor[organ][0] += (x*stride+16)*prob[0][organ].item()
                coor[organ][1] += (y*stride+16)*prob[0][organ].item()
                coor[organ][2] += z*prob[0][organ].item()
                coor[organ][3] += prob[0][organ].item()
            '''
            if guess.item() < 20:
                coor[guess.item()][0] += (x*33+16)*prob[0][guess.item()]
                coor[guess.item()][1] += (y*33+16)*prob[0][guess.item()]
                coor[guess.item()][2] += z*prob[0][guess.item()]
                coor[guess.item()][3] += prob[0][guess.item()]
            '''
    return coor

if __name__ == '__main__':
    stride = int(input("Entrer le stride (entre 1 et 33) : "))
    print(f"stride = {stride}\nimagette par slice = {floor(1+(512-33)/stride)}\npixels ignorés = {(512-33)%stride}\n")

    device = torch.device('cpu')

    
        #else:
        #    if patients_code[int(center_file.split("_")[0])] in fold_to_patients[fold]:
        #        print("skipped :", center_file)
    #for i in patients_centers[0]:
    #    print("\n", i)

    for fold in range(1):
        obj = os.scandir("..\..\AnatomicalStructuresDetector\data\CTce_ThAb")
        patients3D_files = [i.name for i in obj]
        patients3D_files = [patients3D_files[i] for i in fold_to_patients[fold]]
        # patients3D_files = patients3D_files

        obj = os.scandir("..\..\AnatomicalStructuresDetector\data\centers\centers")
        center_files = [i.name for i in obj]
        #center_files = [center_files[i] for i in fold_to_patients[fold]]
        patients_centers = [[0]*20 for i in range(5)]
        for center_file in center_files:
            if int(center_file.split("_")[4]) not in [0, 1, 2] and patients_code[int(center_file.split("_")[0])] in fold_to_patients[fold]: # pas les classes artificelles
                patients_centers[group_code[patients_code[int(center_file.split("_")[0])]]][organs_code[int(center_file.split("_")[4])]] = center_file  

        fold_time = time.time()
        print(f"Fold {fold}")
        for patient in range(1):
            patient_time = time.time()
            print(f"Patient {patients3D_files[patient]}\n-------------------------------")
            model = Net()
            model.load_state_dict(torch.load("..\..\AnatomicalStructuresDetector\models\\fold"+str(fold)+".pt", map_location=device))
            model.eval()
            
            epi_img = nib.load("..\..\AnatomicalStructuresDetector\data\CTce_ThAb\\"+patients3D_files[patient])
            epi_img_data = epi_img.get_fdata()

            with Pool() as p:
                res = p.starmap(z_slice, [(epi_img_data[:,:,z], model, z, stride) for z in range(200, 201)]) #epi_img_data.shape[2]
            coor = [[0, 0, 0, 0] for i in range(20)]

            for slice in range(len(res)):
                for organ in range(20):
                    coor[organ][0] += res[slice][organ][0]
                    coor[organ][1] += res[slice][organ][1]
                    coor[organ][2] += res[slice][organ][2]
                    coor[organ][3] += res[slice][organ][3]

            #for organ in range(20):
                #print(f"Organ {organ} : {coor[organ][3]}")

            '''
            for patient in range(1): #len(patients3D_files)
                print(f"Patient {patients3D_files[patient]}\n-------------------------------")
                coor = [[0, 0, 0, 0] for i in range(20)]

                epi_img = nib.load("data/CTce_ThAb/"+patients3D_files[patient])
                epi_img_data = epi_img.get_fdata()
                
                for z in range(epi_img_data.shape[2]):
                    print(f"Slice {z}")
                    coor = z_slice(epi_img_data[:,:,z], coor)
            '''
            distance = np.array([0, 0])
            for organ in range(20):
                if patients_centers[patient][organ] != 0:
                    f = open("..\..\AnatomicalStructuresDetector\data\centers\centers\\"+patients_centers[patient][organ])
                    csvreader = csv.reader(f)
                    for z in csvreader:
                        real_coor = list(map(float,z))
                    p = real_coor[3]/real_coor[0]
                    q = real_coor[4]/real_coor[1]
                    r = real_coor[5]/real_coor[2]
                    mat_passage = np.diagflat((p, q, r), k=0)
                    vox_coor = np.transpose(coor[organ][:-1], axes=None)
                    obj_coor = np.dot(mat_passage, vox_coor)
                    x = round(obj_coor[0]/coor[organ][3], 2)
                    y = round(obj_coor[1]/coor[organ][3], 2)
                    z = round(obj_coor[2]/coor[organ][3], 2)
                    euclidian = dist((x,y,z),(real_coor[3],real_coor[4],real_coor[5]))
                    print(f"{organs[list(organs_code.keys())[list(organs_code.values()).index(organ)]]} : ({x},{y},{z}), Real Center : {round(real_coor[3], 3), round(real_coor[4], 3), round(real_coor[5], 3)}, d = {round(euclidian ,2)}")
                    distance = np.vstack((distance, np.array([organ, euclidian]))) #organ , dist((x,y,z),(real_coor[3],real_coor[4],real_coor[5]))
            np.savetxt("patient"+str(patient)+"_dist.csv", distance, fmt='%5f', delimiter=';')
            print(f"Patient {patient} done in {time.time()-patient_time} seconds")
        print(f"Fold {fold} done in {time.time()-fold_time} seconds")
''' DRAW 
image = np.array(epi_img_data[:, 280, :])
for x in image:
    for z in range(epi_img_data.shape[2]):
        x[z] = translate(x[z])
image = Image.fromarray(np.asarray(image, dtype='uint8'))
filename = f"CentersResultFold{1}.png"
image.save(filename)
input_image = Image.open(filename)
input_image = input_image.convert('RGB')
draw = ImageDraw.Draw(input_image)
colors = [  
            "red", "green", "blue", "yellow",
            "purple", "orange", "pink", "maroon",
            "white", "limegreen", "magenta", "lightsalmon",
            "mediumblue", "olivedrab", "peru", "sienna",
            "violet", "burlywood", "aqua", "goldenrod",
            "navy", "oldlace", "moccasin"
         ]
a = input_image.size
print(a)
for i,color in zip(range(20),colors):
    draw.line([(coor[i][2]-6,coor[i][0]-6),(coor[i][2]+6,coor[i][0]+6)], width=3, 
                fill=color)
    draw.line([(coor[i][2]+6,coor[i][0]-6),(coor[i][2]-6,coor[i][0]+6)], width=3, 
                fill=color)
input_image = input_image.transpose(Image.ROTATE_90)
input_image.save(filename)
'''
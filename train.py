# %%
import copy
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

# %%
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model,Blip2Config
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score

# %%
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from model.models import ModelParallelTemporalQformer
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description='训练模型')

# 添加参数
parser.add_argument('--dataset_root', type=str, default='data/train', help='training data path')
parser.add_argument('--val_root', type=str, default='data/val', help='validation data path')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--warmup_epochs', type=int, default=6, help='warmup epochs')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')


args = parser.parse_args()

dataset_root = args.dataset_root
val_root = args.val_root
initial_lr = args.lr
warmup_epochs = args.warmup_epochs
num_epochs = args.num_epochs
batch_size = args.batch_size


def get_type(lines):
    #  with open(file_single) as f:
    #     lines = [line.rstrip('\n') for line in f]
        line = lines[0]
        #scenario_length_list.append(int(line.split(' ')[-1])-10)
        #weather_name_list.append(line.split(' ')[0])
        # if line.split(' ')[1] != '-1':
        #     print(file_single.split('/')[-1])
        #crash_list.append(line.split(' ')[1] != '-1')
        #subtype_name = file_single.split('/')[-1].split('_')[2]
        road_type = lines[3].split(': ')[-1]
        spawn_side = lines[4].split(': ')[-1]
        ego_direction = lines[5].split(': ')[-1]
        other_direction = lines[6].split(': ')[-1]

        accident_id1, accident_cls1, accident_id2, accident_cls2 = int(lines[0].split(' ')[1]), lines[0].split(' ')[2], int(lines[0].split(' ')[3]), lines[0].split(' ')[4]
        designed_ids_str = lines[2].split(': ')[-1].split(' ')
        designed_ids = [int(designed_ids_str[0]), int(designed_ids_str[1]), int(designed_ids_str[2]), int(designed_ids_str[3])]

        scenario_type = 'None'
        if accident_id1 == -1 or accident_id2 == -1:
            scenario_type = 'no accident'
        elif accident_cls1 == 'pedestrian' or accident_cls2 == 'pedestrian':
            scenario_type = 'collide with pedestrians'
        elif accident_id1 not in designed_ids or accident_id2 not in designed_ids:
            scenario_type = 'collide with other vehicles'
        else:
            if road_type == 'four-way junction':
                if ego_direction == 'straight' and other_direction == 'straight' and spawn_side != 'opposite':
                    scenario_type =  'straight straight'
                elif spawn_side != 'opposite' and ((ego_direction == 'straight' and other_direction == 'left') or (ego_direction == 'left' and other_direction == 'straight')):
                    scenario_type =  'straight left side'
                elif spawn_side == 'opposite' and ((ego_direction == 'straight' and other_direction == 'left') or (ego_direction == 'left' and other_direction == 'straight')):
                    scenario_type = 'straight left opposite'
                elif spawn_side == 'opposite' and ((ego_direction == 'left' and other_direction == 'right') or (ego_direction == 'right' and other_direction == 'left')):
                    scenario_type = 'left right opposite'
            else:
                if ego_direction == 'straight' or other_direction == 'straight':
                    scenario_type = 'three-way straight'
                else:
                    scenario_type = 'three-way both turns'
        scenario_types = ['no accident', 'collide with pedestrians', 'collide with other vehicles',
                      'straight straight', 'straight left side', 'straight left opposite',
                      'left right opposite', 'three-way straight', 'three-way both turns']
        scenario_type_idx = scenario_types.index(scenario_type)


        #scenario_type_one_hot = F.one_hot(torch.tensor(scenario_type_idx), len(scenario_types)).float()
        #true_dist = smooth_one_hot(scenario_type_one_hot, classes=5, smoothing=0.1)

        return scenario_types[scenario_type_idx]

class ResizeAndPad(nn.Module):
    def __init__(self, output_size):
        super(ResizeAndPad, self).__init__()
        self.output_size = output_size

    def forward(self, image):
        old_width, old_height = image.size
        new_width, new_height = self.output_size
        
        # 计算缩放比例
        scale = min(new_width / old_width, new_height / old_height)
        new_size = (int(old_width * scale), int(old_height * scale))
        
        # 缩放图像
        image = transforms.Resize(new_size)(image)
        
        # 创建目标图像，填充成 224x224
        pad_width = new_width - new_size[0]
        pad_height = new_height - new_size[1]
        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
        padded_image = transforms.functional.pad(image, padding, fill=0, padding_mode='constant')
        
        return padded_image

class CustomDataset(Dataset):
    def __init__(self, root_dir, accident_folders, normal_folders):
        self.root_dir = root_dir
        self.accident_folder = accident_folders  # ['type1_subtype1_accident', 'type1_subtype2_accident']
        self.normal_folder = normal_folders  # ['type1_subtype1_normal','type1_subtype2_normal']
        self.viewpoints = ['Camera_Front', 'Camera_FrontLeft', 'Camera_FrontRight', 'Camera_Back', 'Camera_BackLeft',
                            'Camera_BackRight']

        self.vehicles = ['ego_vehicle','ego_vehicle_behind','other_vehicle','other_vehicle_behind']

        self.data = self.build_dataset()

    def build_dataset(self,is_predict = True):
        data = []
        for category in self.accident_folder + self.normal_folder:
            category_path = os.path.join(self.root_dir, category)
            scenario_folders = os.listdir(os.path.join(category_path, "ego_vehicle",self.viewpoints[0]))
            #print(scenario_folders)
            #print("scenario",len(scenario_folders))
            for scenario_folder in scenario_folders:
                scenario_path = os.path.join(category_path,"ego_vehicle", self.viewpoints[0], scenario_folder)
                image_files = sorted(os.listdir(scenario_path))
                scenario_images = []
                count = 0
                # print(type(image_files))
                if len(image_files) < 100:
                    for j in range(100 - len(image_files)):
                        image_files.insert(0, image_files[0])

    #    scenario_type_idx = scenario_types.index(scenario_type)


    #     scenario_type_one_hot = F.one_hot(torch.tensor(scenario_type_idx), len(scenario_types)).float()
    #     #true_dist = smooth_one_hot(scenario_type_one_hot, classes=5, smoothing=0.1)

                # get meta
                meta_path = os.path.join(category_path, 'meta', scenario_folder+".txt")
                        
                with open(meta_path) as f:
                        lines = [line.rstrip('\n') for line in f]
                type_meta = get_type(lines)

                meta_one_hot = None
                if is_predict == True:
                    if type_meta == "no accident":
                        meta_one_hot = F.one_hot(torch.tensor(0), 2).float()
                    else:
                        meta_one_hot = F.one_hot(torch.tensor(1), 2).float()
                else:
                    accident_types = [
                      'straight straight', 'straight left side', 'straight left opposite',
                      'left right opposite', 'three-way straight', 'three-way both turns']
                    if type_meta not in accident_types:
                        continue
                    else:
                        scenario_type_idx = scenario_types.index(accident_types)


                        meta_one_hot = F.one_hot(torch.tensor(scenario_type_idx), len(accident_types)).float()


                for i in range(len(image_files)):
                    
                    count += 1
                    vehicle_images = []
                    for vehicle in self.vehicles:
                   
                        # get viewpoints images
                        viewpoint_images = []
                        for viewpoint in self.viewpoints:
                            image_path = os.path.join(category_path, vehicle, viewpoint, scenario_folder, image_files[i])
                            viewpoint_images.append(image_path)  # Store paths instead of loading images
                        
                        vehicle_images.append(viewpoint_images) # [6,3,224,224]
                    #temporal_images.append(vehicle_images) # [4, 6, 3, 224, 224]
                    scenario_images.append(vehicle_images) # [100, 4, 6, 3, 224, 224]
                #print(scenario_images)
                data.append([category, scenario_images,meta_one_hot])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category, scenario_image_paths,type_meta = self.data[idx]
        #print('cringe', len(scenario_image_paths))
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            #ResizeAndPad((224,224)),
            transforms.ToTensor()  # 将数据转换为PyTorch tensor
        ])
        batch_images = []
        for timestep_image_paths in scenario_image_paths:
            vehicle_image = []
            for vehicle in timestep_image_paths:
                
                images = [transform(Image.open(image_path).convert('RGB')) for image_path in vehicle]
                vehicle_image.append(torch.stack(images))
            
            batch_images.append(torch.stack(vehicle_image))
        batch_images = torch.stack(batch_images)
        # Label: 1 for accident, 0 for normal
        # print(str(scenario_images[0][0]))
        label = [0, 1] if 'accident' in category else [1, 0]
        label = torch.Tensor(label)
        return batch_images, type_meta

# %%
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
     




    def forward(self, x):
        #x = x.view(x.size(0), -1)  # 将输入展平成一维向量，适应线性层的输入
        x = self.linear1(x)
        #x = self.linear2(x)


        return x



custom_dataset = CustomDataset(dataset_root,['type1_subtype1_accident','type1_subtype2_accident'],['type1_subtype1_normal','type1_subtype2_normal'])
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = CustomDataset(val_root,['type1_subtype1_accident','type1_subtype2_accident'],['type1_subtype1_normal','type1_subtype2_normal'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

device = "cuda" 

model = ModelParallelTemporalQformer()

model.qformer1.vision_model.requires_grad = False
model.qformer2.vision_model.requires_grad = False
model.qformer3.vision_model.requires_grad = False
model.qformer4.vision_model.requires_grad = False

model.to(device)
torch.save(model.state_dict(), './test.pt')
criterion = torch.nn.CrossEntropyLoss()

warmup_factor = 1.0 / warmup_epochs
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_factor * min(1.0, epoch * 1.0 / warmup_epochs))

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

# 训练循环
for epoch in range(num_epochs):
    model.train()
    
    total_predictions = []
    total_labels = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    
    for batch_idx, (inputs, labels_one_hot) in progress_bar:
        inputs, labels_one_hot = inputs.to(device), labels_one_hot.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_one_hot)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        labels = torch.argmax(labels_one_hot, dim=1)
        total_predictions.extend(predicted.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'Loss': loss.item()})
    
    epoch_acc = accuracy_score(total_labels, total_predictions)
    epoch_recall = recall_score(total_labels, total_predictions, average='weighted')
    print(f'Epoch [{epoch+1}/{num_epochs}], TrainAcc: {epoch_acc:.4f}, TrainRecall: {epoch_recall:.4f}')
    scheduler.step()
    
    # 验证循环
    model.eval()
    total_predictions = []
    total_labels = []
    with torch.no_grad():
        for inputs, labels_one_hot in val_loader:
            inputs, labels_one_hot = inputs.to(device), labels_one_hot.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            labels = torch.argmax(labels_one_hot, dim=1)
            total_predictions.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
      
        epoch_acc = accuracy_score(total_labels, total_predictions)
        epoch_recall = recall_score(total_labels, total_predictions, average='weighted')

        print(f'Epoch [{epoch+1}/{num_epochs}], ValAcc: {epoch_acc:.4f}, ValRecall: {epoch_recall:.4f}')
        
        model_path = f'TemporalFormer_{epoch + 1}_Accuracy_softmax_lr_{initial_lr}_warmup_{warmup_epochs}: {epoch_acc:.3f}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model parameters saved to {model_path}')
from turtle import down
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset,SubsetRandomSampler,DataLoader,Dataset,WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Any


torch.manual_seed(42)
np.random.seed(42)

class Cifar10(Dataset):
    def __init__(self,train_dataset,index,transforms):
        self.index=index
        self.transforms=transforms
        self.train_dataset=train_dataset

    
    def __len__(self):
        return len(self.index)

    def __getitem__(self,idx):
        image,label=self.train_dataset[idx]
        if self.transforms:
            transformed_image=self.transforms(image)
        else:
            transformed_image=image
        return transformed_image,label
        




class NeuraMatrixDataset():
    def __init__(self,val_size,transform=None,rand_state=42,batch_size=32):
        self.doc=[''' 
            Data set for the event of this symposium Avinyaa 24 this uses the CIFAR-10 Dataset\n
            Which is loaded from the torchvision.datasets.CIFAR10 and with the random state.\n
            we further more split the training and testing size on to the user's Wish \n
            and Data transformaions techniques could also been left to users to do\n
            or it is uses tensor and normalization on Imagenet weights by default\n
            The torch is prefered If you are using albumentations Write them as a seperate function\n 
            and put them in the transform\n''',
            '''Parameters:\n
            val_size: the size of the validation/dev set\n
            rand_state: custum random state default 42\n''',

            '''Methods\n
            get_train_validation_dataset\n
            '''

            '''Returns \n
            Training_data,testing_data\n
            ''']
        for i in self.doc:
            for  j in i.split('\n'):
                print(j,sep='\n')
        self.test_size=val_size
        self.rand_state=rand_state
        self.train_dataset=torchvision.datasets.CIFAR10(
            root='./data',train=True,download=True)
        self.test_dataset=torchvision.datasets.CIFAR10(
            root='./data',train=False,download=True)
        
        self.transforms=transform
        self.batch_size=batch_size

    def get_sampling_weights(self,dataloader):
        train_d={}
        targets=[]
        for i in dataloader:
            image,label=i
            temp_label=list(label.detach().cpu().numpy())
            targets.extend(temp_label)
            count=set(temp_label)
            for i in count:
                if not i in train_d.keys():
                    train_d[i]=temp_label.count(i)
                else:
                    train_d[i]+=temp_label.count(i)
        
        sample_count=np.array([train_d[i] for i in train_d.keys()])
        sample_weights=1./sample_count

        total_sample_weights=sample_weights[targets]
        sample_weight_tensor=torch.from_numpy(total_sample_weights)
        return WeightedRandomSampler(
            weights=sample_weight_tensor, # type: ignore
            num_samples=len(total_sample_weights),
            replacement=True,

        )
    
    def get_train_validation_dataset(self):
        train,test=train_test_split(np.arange(0,len(self.train_dataset)),test_size=self.test_size,random_state=self.rand_state)
        train_sampler=self.get_sampling_weights(DataLoader(Cifar10(self.train_dataset,list(train),self.transforms),batch_size=self.batch_size))
        test_sampler=self.get_sampling_weights(
            DataLoader(Cifar10(self.train_dataset,list(test),self.transforms),batch_size=self.batch_size)
        )
        train_dataloader=DataLoader(Cifar10(self.train_dataset,list(train),self.transforms),batch_size=self.batch_size,shuffle=False,
                                    sampler=train_sampler)
        test_dataloader=DataLoader(Cifar10(self.train_dataset,list(test),self.transforms),batch_size=self.batch_size,shuffle=False,
                                   sampler=test_sampler)
        return train_dataloader,test_dataloader

    def get_test_dataset(self):
        test_sampler=self.get_sampling_weights(DataLoader(Cifar10(
            self.test_dataset,list(np.arange(0,len(self.test_dataset))),transforms=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
                ]
            )),batch_size=self.batch_size))
        
        test_dataloader=DataLoader(Cifar10(self.test_dataset,list(np.arange(0,len(self.test_dataset))),transforms=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
                ]
            )),batch_size=self.batch_size,shuffle=False,sampler=test_sampler)
        
        return test_dataloader
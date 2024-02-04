from AvDataloader import NeuraMatrixDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision
from tqdm.notebook import tqdm
from torchvision import transforms
import torch.nn as nn
from torchmetrics import Accuracy
import torch.nn.functional as F
import os

def iter_through_traindataset(dataset,model,device,Accuracy_fn,loss_fn,optimizer):
    running_loss=[]
    running_accuracy=[]
    model.train()
    for i in tqdm(dataset,total=len(dataset)):
        optimizer.zero_grad()
        image,label=i
        image=image.to(device)
        label=F.one_hot(label.long(),num_classes=10).float()
        y_pred=model(image)

        y_pred_labels=y_pred.argmax(dim=1)
        
        loss=loss_fn(y_pred,label)
        
        loss.backward()

        accuracy=Accuracy_fn(
            label.argmax(dim=1),
            y_pred_labels
        )

        optimizer.step()

        running_loss.append(loss.item())
        running_accuracy.append(accuracy.item())

    return np.mean(running_loss),np.mean(running_accuracy)        

def iter_through_valdataset(dataset,model,device,Accuracy_fn,loss_fn):
    with torch.no_grad():
        running_loss=[]
        running_accuracy=[]
        model.eval()
        for i in tqdm(dataset,total=len(dataset)):
            image,label=i
            image=image.to(device)
            label=F.one_hot(label.long(),num_classes=10).float()

            y_pred=model(image)

            y_pred_labels=y_pred.argmax(dim=1)
            loss=loss_fn(y_pred,label)

            accuracy=Accuracy_fn(
            label.argmax(dim=1),
            y_pred_labels
            )

            running_loss.append(loss.item())
            running_accuracy.append(accuracy.item())

        return np.mean(running_loss),np.mean(running_accuracy)
    
def training_model(model,loss_fn,device,accuracy_fn,train_data,val_data,epochs,lr=1e-3):
    history={
        
    }
    model=model.to(device)
    best_val=1000000.0
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    os.makedirs("Modelweights")
    for epoch in range(epochs):
        print(f" Epoch: {epoch+1} \n")
        train_loss,train_acc=iter_through_traindataset(train_data,model,device,accuracy_fn,loss_fn,optimizer)
        
        if not 'train_loss' in history.keys():
            history['train_loss']=[train_loss]
        else:
            history['train_loss'].append(train_loss)
        
        if not 'train_acc' in history.keys():
            history['train_acc']=[train_acc]
        else:
            history['train_acc'].append(train_acc)

        val_loss,val_acc=iter_through_valdataset(val_data,model=model,device=device,Accuracy_fn=accuracy_fn,loss_fn=loss_fn)
        
        if not 'val_loss' in history.keys():
            history['val_loss']=[val_loss]
        else:
            history['val_loss'].append(val_loss)
        
        if not 'val_acc' in history.keys():
            history['val_acc']=[val_acc]
        else:
            history['val_acc'].append(val_acc)
        

        if val_loss < best_val: # type: ignore
            print("best validation loss is found")
            torch.save(model.state_dict(),"Modelweights/best_model.pth")


        print(f"""
                    train_loss {train_loss},
                    val_loss {val_loss},\n
                    train_acc {train_acc},
                    val_acc {val_acc}""")
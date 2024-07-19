import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import random

from torchvision import transforms
from datetime import datetime

from loss_function import FocalLoss, TverskyLoss, FocalTverskyLoss,BinaryDiceLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataloader import ISLES2018Dataset
from sklearn.model_selection import KFold

from lion_pytorch import Lion
'''
from m_transunet import TransUnet
from m_fcn import FCN8s
from m_deeplab import DeepLabV3
#from m_unet import Unet
from m_DeepTransUnet import DeepTransUnet
'''
#from ba_swinTAUNet import swinTAUNet
from ba_TAU_module import TAU_module
seed = random.randint(1,100)
print(f"seed id {seed}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)
#training_part, testing_part = torch.utils.data.random_split(dataset_m, (480,22), generator=torch.Generator().manual_seed(seed)) # random_split(数据集，长度)随机将一个数据集分割成给定长度的不重叠的新数据集
training_part, testing_part = torch.utils.data.random_split(dataset_m, (480,22), generator=torch.Generator().manual_seed(seed)) # random_split(数据集，长度)随机将一个数据集分割成给定长度的不重叠的新数据集


# weight_init
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# Configuration options
#k_folds = 5
k_folds = 5
# For fold results
results = {}

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

def train_model():
    max_acc = 0.0000
    max_epoch = 0
    # Start print
    print('-'*50)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(training_part)):

        # 创建csv文件
        df = pd.DataFrame(columns=['Time', 'Step', 'Train loss', 'Validation loss', 'Validation dice'])# 列名
        df.to_csv(f'ba_swinTAUNet_300-fold-{fold}.csv',index=False)  

        # Print
        print(f'FOLD {fold}')
        print('-'*50)

        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(training_part,
                                batch_size=6,
                                sampler=train_subsampler)
        
        validationloader = DataLoader(training_part,
                                      batch_size=4,
                                      sampler=validation_subsampler)
        print(len(trainloader))
        print(len(validationloader))

        # Init the network
        model = TAU_module()
        #model = DeepTransUnet()
        #加载权重继续训练
        #model.load_state_dict(torch.load('ba_swinTNet.pth'))

        model.apply(weight_init)
        model.to(device)
        binarydice_loss = BinaryDiceLoss()
        bce_loss = nn.BCELoss()
        focal_loss = FocalLoss()
        focaltversky_loss = FocalTverskyLoss()
        #optimizer = Lion(model.parameters(), lr=0.0000523)
        #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000623, weight_decay=0.05)#0000523
        #optimizer = torch.optim.NAdam(model.parameters(), lr=0.0000623, weight_decay=0.05)#0000523
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.000623, weight_decay=0.05)#0000523
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0000623, weight_decay=0.05)#0000523
        #optimizer = torch.optim.Adamax(model.parameters(), lr=0.0000623, weight_decay=0.05)#0000523
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.8,patience=10) # 3, 0.6

        for epoch in range(300):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # 获取当前时间
            Time = "%s"%datetime.now()
            Step ="Step[%d]"%(epoch+1)

            ### 训练 ###
            model.train() 

            training_loss = 0.0
        
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                modalities, OT = data[0].to(device), data[1].to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                output = model(modalities)

                # Compute loss
                loss, _ = binarydice_loss(output, OT)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                training_loss += loss.item()

            print('Train Loss : {:.3f}'.format(training_loss/len(trainloader)))
            
            # Process is complete
            print('Training process has finished!')

            # Print validation
            print('Starting validation')

            # Saving the model
            save_path = f'ba_swinTAUNet_300-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            train_loss = training_loss/len(trainloader)
            Train_loss = "%f"%train_loss

            ### 评估 ###
            model.eval()
            
            # Evaluation for this fold
            with torch.no_grad():

                val_loss = 0.0
                val_dice = 0.0

                # Iterate over the test data and generate predictions
                for i, data in enumerate(validationloader, 0):

                    # Get inputs
                    modalities, OT = data[0].to(device), data[1].to(device)

                    # Generate outputs
                    output = model(modalities)

                    loss, dice = binarydice_loss(output, OT)

                    val_loss += loss.item()
                    val_dice += dice.item()
                
                print('Validation Loss: {:.3f}'.format(val_loss/len(validationloader)))
                print("learning rate : ",optimizer.state_dict()['param_groups'][0]['lr'])
                print('-'*30)

                # Print accuracy
                print('Accuracy for fold %d: %.3f %%' % (fold, 100.0 * (val_dice/len(validationloader))))
                print('-'*30)
                results[fold] = 100.0 * (val_dice/len(validationloader))
                acc = 100.0 * (val_dice/len(validationloader))
                #保存结果最好的那个
                if max_acc < float('%.3f' % acc):
                    max_acc = float('%.3f' % acc)
                    max_epoch = epoch
                    print("save!")
                    # 保存模型语句
                    torch.save(model.state_dict(),f"ba_swinTAUNet_best_{fold}.pth")

                validation_loss = val_loss/len(validationloader)
                Validation_loss = "%f"%validation_loss
                validation_dice = val_dice/len(validationloader)
                Validation_dice = "%f"%validation_dice
               
                scheduler.step(validation_loss)


            #将数据保存为一维列表
            list = [Time, Step, Train_loss, Validation_loss, Validation_dice]
            file = pd.DataFrame([list])
            file.to_csv(f'ba_swinTAUNet_300-fold-{fold}.csv', mode='a', header=False, index=False)
        #记录模型最好的信息
        with open(f"best_{fold}.txt","w") as f:
            f.write(f"fold is {fold},best epoch is {max_epoch}, best dice is {max_acc},seed is {seed}") 
        max_acc = 0.0000
        max_epoch = 0
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('-'*50)
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')



if __name__ == '__main__':
    train_model()
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
import shutil
from tqdm import tqdm
from torchvision import transforms
from skimage import io


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):

    device = torch.device("cuda:0"
                          if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"[INFO] Torch device is {torch.cuda.current_device()}")

    # os.environ['CUDA_VISIBLE_DEVICES']='0'

    DCE_net = model.enhance_net_nopool().to(device)

    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    best_loss = 1e10
    for epoch in range(1, config.num_epochs+1):
        # print(f"[INFO] Epoch [{epoch}/{config.num_epochs}]....")
        tqdm_object = tqdm(
            train_loader, desc=f"[INFO] Epoch [{epoch}/{config.num_epochs}]", total=len(train_loader))
        iteration=1
        loss_epoch = []
        for img_lowlight in tqdm_object:

            img_lowlight = img_lowlight.to(device)

            enhanced_image_1,enhanced_image,A = DCE_net(img_lowlight)

            Loss_TV = 200*L_TV(A)
            
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

            loss_col = 5*torch.mean(L_color(enhanced_image))

            loss_exp = 10*torch.mean(L_exp(enhanced_image))
            
            
            # best_loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp
            #

            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
            optimizer.step()

            # if ((iteration) % config.display_iter) == 0:
            #     print(">>>> Loss at iteration", iteration, ":", loss.item())
            if ((iteration) % config.checkpoint_iter) == 0:
                torch.save(DCE_net.state_dict(), config.checkpoints_folder + "Epoch" + str(epoch) + '.pth') 
            loss_epoch.append(loss.item())
            iteration += 1
        loss_epoch = sum(loss_epoch)/len(loss_epoch)
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(DCE_net.state_dict(), config.checkpoints_folder + "best_model.pth")
            print(">>>> Best model saved!!!!")
        print(f">>>> Epoch [{epoch}/{config.num_epochs}]: Loss: {loss_epoch} - Best Loss: {best_loss}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="/home/ubuntu/thanh.nt176874/dangnh/Object-Detection-In-Night-Vision/Dataset/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--checkpoint_iter', type=int, default=10)
    parser.add_argument('--checkpoints_folder', type=str, default="checkpoints/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "checkpoints/Epoch99.pth")

    config = parser.parse_args()

    if os.path.exists(config.checkpoints_folder):
        shutil.rmtree(config.checkpoints_folder)
    os.mkdir(config.checkpoints_folder)

    train(config)








	

#I want a job so bad((
#
import torch
import torch.nn as nn
import torch.nn.functional as func
#
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) #a little fix for colab
#

#I launch it only with Google colab because I don't even have GPU, this learning will immediately kill my pc
RESNET_BLOCKS = 10
CHANNELS = 128
#


##########################
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, tensor): #no step back little pawn!
        residual = tensor
        tensor = func.relu(self.bn1(self.conv1(tensor))) #to conv to bn to relu
        tensor = self.bn2(self.conv2(tensor)) #to conv to bin only

        return func.relu(tensor + residual)
##########################


##########################
class ChessNet(nn.Module):
    def __init__(self, blocks = RESNET_BLOCKS, channels = CHANNELS):
        super().__init__()

        #previously we had (18, 8, 8) tensor, so now we turn 18 to channels
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        #stack of 10 ResBlocks
        self.tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

        #next move prediction and rules
        self.pred_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.pred_bn = nn.BatchNorm2d(32)
        self.pred_fc = nn.Linear(32 * 8 * 8, 4096)

        #winner prediction
        self.win_conv = nn.Conv2d(channels, 4, 1, bias = False)
        self.win_bn = nn.BatchNorm2d(4)
        self.win_fc1 = nn.Linear(4 * 8 * 8, 64)
        self.win_fc2 = nn.Linear(64, 1)


    #Polymorphism in Python refers to the ability of different objects to respond 
    #to the same method or function call in ways specific to their individual types. 
    #The term comes from Greek words “poly” (many) and “morphos” (forms), literally meaning “many forms.” 
    #Polymorphism is a core concept in object-oriented programming (OOP) 
    #that allows programmers to use a single interface with different underlying forms.
    def forward(self, tensor):
        tensor = self.stem(tensor)
        tensor = self.tower(tensor)

        #prediction
        p = func.relu(self.pred_bn(self.pred_conv(tensor))) #to conv to bn to relu!
        p = p.view(p.size(0), -1)
        prediction = self.pred_fc(p)

        #winner
        w = func.relu(self.win_bn(self.win_conv(tensor))) #to conv to bn to relu!
        w = w.view(w.size(0), -1)
        w = func.relu(self.win_fc1(w))
        winner = torch.tanh(self.win_fc2(w))

        return prediction, winner
##########################


###############################################
def save_model(model, path="weights/model.pt"):
    torch.save(model.state_dict(), path)
###############################################


#################################################################################
def load_model(path="weights/model.pt", blocks=RESNET_BLOCKS, channels=CHANNELS):
    model = ChessNet(blocks=blocks, channels=channels)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
#################################################################################

#The Rectified Linear Unit (ReLU) is a widely used activation function in deep learning, 
#defined as f(x) = max(0, x). It outputs the input directly if positive, and zero otherwise. 
#ReLU introduces non-linearity, allowing networks to learn complex patterns, 
#while accelerating training due to its computational simplicity
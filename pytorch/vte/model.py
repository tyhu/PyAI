import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgBranch(nn.Module):
    def __init__(self):
        super(ImgBranch, self).__init__()
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.nb2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.nb3 = nn.BatchNorm1d(512)

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x_res = F.relu(self.nb2(self.fc2(x)))
        x = x_res + x
        x_res = F.relu(self.nb3(self.fc3(x)))
        x = x_res + x
        return x

class TextBranch(nn.Module):
    def __init__(self):
        super(TextBranch, self).__init__()
        self.fc1 = nn.Linear(in_features=18000, out_features=2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=2000)
        self.nb2 = nn.BatchNorm1d(2000)
        self.fc3 = nn.Linear(in_features=2000, out_features=512)
        self.nb3 = nn.BatchNorm1d(512)

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x_res = F.relu(self.nb2(self.fc2(x)))
        x = x_res + x
        x = F.relu(self.nb3(self.fc3(x)))
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        sim1 = torch.sum(anchor*positive,1)
        sim2 = torch.sum(anchor*negative,1)
        #print sim1
        #print sim2
        dist = sim2-sim1+self.margin
        dist_hinge = torch.clamp(dist, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class ImgBranch2(nn.Module):
    def __init__(self):
        super(ImgBranch2, self).__init__()
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.nb = nn.BatchNorm1d(512)
        self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.dp(F.relu(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.nb(self.fc2(x))
        norm = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(norm)
        return x

class TextBranch2(nn.Module):
    def __init__(self):
        super(TextBranch2, self).__init__()
        self.fc1 = nn.Linear(in_features=6000, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.nb = nn.BatchNorm1d(512)
        self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.dp(F.relu(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.nb(self.fc2(x))
        norm = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(norm)
        return x


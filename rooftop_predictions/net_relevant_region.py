import torch
import torch.nn as nn
class Relevant_Region(nn.Module):
    def __init__(self, n_df=8, n_cnv=3, img_size=(256, 256), p_dropout=0.7):
        super(Relevant_Region, self).__init__()
        self.bias = False
        self.n_df = n_df
        self.p_dropout = p_dropout
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(2)
        self.n_cnv = n_cnv
        self.n_cnvt = n_cnv
        self.cnv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        # self.cnv_layers2 = nn.ModuleList()
        self.cnvt_layers = nn.ModuleList()
        self.bn_layers_cnvt = nn.ModuleList()
        for k in range(self.n_cnv):
            if k == 0:
                n_in = 3
                n_out = n_df
            else:
                n_in = 2**(k-1)*n_df
                n_out = 2**k*n_df
            # self.cnv_layers.append(nn.Conv2d(n_in, n_out, 3, 1, 1, bias=self.bias))
            # self.cnv_layers2.append(nn.Conv2d(n_out, n_out, 3, 1, 1, bias=self.bias))
            self.cnv_layers.append(nn.Conv2d(n_in, n_out, 4, 2, 1, bias=self.bias))
            self.bn_layers.append(nn.BatchNorm2d(n_out))
        for k in range(self.n_cnvt):
            if k == self.n_cnvt - 1:
                n_in = 3
            self.cnvt_layers.append(nn.ConvTranspose2d(n_out, n_in, 4, 2, 1, bias=self.bias))
            # self.bn_layers_cnvt.append(nn.BatchNorm2d(n_in))
            n_in = n_in // 2
            n_out = n_out // 2

    def forward(self, x):
        for k, _ in enumerate(range(self.n_cnv)):
            x = self.cnv_layers[k](x)
            # x = self.lrelu(x)
            # x = self.cnv_layers2[k](x)
            x = self.bn_layers[k](x)
            x = self.relu(x)
            # if k == 2:
            #   x1 = x
            # x = self.maxpool(x)
            # if k == self.n_cnv - 1 and dropout:
            #     x = self.dropout(x)
        # x = self.avg_pool(x)
        for k, _ in enumerate(range(self.n_cnvt)):
            x = self.cnvt_layers[k](x)
            # x = self.bn_layers_cnvt[k](x)
            x = self.relu(x)
            if k == 0:
                x = self.dropout(x)
            # if k == 1:
            #   # print(x.size())
            #   # print(x1.size())
            #   x += x1
            # if dropout:
            #     x = self.dropout(x)
        x = self.tanh(x)**2
        # x = self.tanh(x)
        # x = self.relu(x)
        return x
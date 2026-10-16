import torch
from torch import nn

import numpy as np

class PPEG(nn.Module):
    def __init__(self, dim=256,k=7,conv_1d=False,bias=True):#dim与切片特征保持一致
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)#分层卷积，1为步伐，k//2为扩充
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)#在没有批次维度的情况下，增加一个批次维度
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))#对N求取根号，并向上取整
        
        add_length = H * W - N#补充的切片数量
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) #这里复制是前几个token，这样切片之间分布比较均匀

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)#防止切片的两倍依然不够
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device,dtype=x.dtype)],dim = 1)
            add_length += zero_pad#增加值计数

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W).contiguous()#这里换位加形状重塑，从切片数量维度转为正方形维度
        #cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)#进行卷积相加
        x = x.flatten(2).transpose(1, 2)#将正方形展平会原来的切片序列格式
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]#同时去除增加的维度，保持和原始特征一样的格式
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class SINCOS(nn.Module):
    def __init__(self):
        super(SINCOS, self).__init__()
        #self.embed_dim = embed_dim
        # self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)

    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos, device):#输入特征维度，输入一维坐标序列，生成一维的位置编码
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float, device=device)
        omega = omega / (embed_dim / 2.)
        omega = 1. / (10000**omega)  # (D/2,)#生成频率分量

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid, device):#输入二维坐标（2，H，W），分别取x和y的二维值（1，H，W），然后输入，输入以后展平，变为H*W的一维序列，分别生成编码，再拼接
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], device)  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], device)  # (H*W, D/2)

        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size_h, grid_size_w, device='cpu', cls_token=False):#生成grid，加上调用前一个函数，是一个封装
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """

        grid_h = torch.arange(grid_size_h, dtype=torch.float, device=device)
        grid_w = torch.arange(grid_size_w, dtype=torch.float, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij') 
        grid = torch.stack([grid[1],grid[0]], dim=0)#01代表数值维度，和宽高没关系

        grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device)

        if cls_token:#在给定一个全局信息token时，会产生用处，用于给这个token一个位置表示，但是不需要传达任何位置信息，因为他本身不是一个切片
            pos_embed = torch.cat([torch.zeros(1, embed_dim, device=device), pos_embed], dim=0)
        return pos_embed

    def forward_single(self,x,pos=None,C=None):#输入坐标信息和坐标，然后获取位置编码
        pos_all,pos = pos[0],pos[1:]#这里前者可能是H和W的信息，后者是位置编码

        pos = torch.tensor([ _pos[1]*pos_all[0]+_pos[0] for _pos in pos],device=x.device)#这里是将二维的表示转换为1维的表示

        pos_embed = self.get_2d_sincos_pos_embed(C, int(pos_all[1]),int(pos_all[0]),device=x.device)
        
        pos_embed = pos_embed[pos]

        x = x + pos_embed.unsqueeze(0)

        return x

    def forward(self, x, pos=None):
        B,N,C = x.shape

        if pos.size(0) == 1:
            pos = pos[0]

        if len(pos.shape) == 3:
            for i,_pos in enumerate(pos):
                x[i] = self.forward_single(x[i],_pos,C)
        else:
            x = self.forward_single(x,pos,C)
        
        return x
    
    
    
#代码逻辑：输入x和坐标，然后将坐标转换为一维表示，同时根据坐标的宽高信息，绘制一个二维网格，网格涵盖所有坐标，然后获取对应的序列坐标编码，再用我们坐标的一维索引获取对应的编码，再与x拼接。x维度是bnc，坐标是bn2，坐标的第一个值是宽高信息
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Attention(nn.Module):
    def __init__(self, q_size,k_size,v_size, hid, dropout=0.1):
        super(Attention, self).__init__()
        
        self.hid = hid
        self.fc_Q = nn.Linear(q_size, hid)
        self.fc_K = nn.Linear(k_size, hid)
        self.fc_V = nn.Linear(v_size, hid)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(hid, hid)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q,k,v,):
        batch_size = q.size(0)
        Q = self.fc_Q(q)
        K = self.fc_K(k)
        V = self.fc_V(v)
        Q = Q.view(batch_size , -1, self.hid)
        K = K.view(batch_size , -1, self.hid)
        V = V.view(batch_size , -1, self.hid)

        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, self.hid)

        out = self.fc(context)
        out = self.dropout(out)

        return out


class S_Attention(nn.Module):
    def __init__(self, ori_size,q_size,k_size,v_size, hid, dropout=0.1):
        super(S_Attention, self).__init__()
        self.hid = hid
        self.fc_Q = nn.Linear(q_size, hid)
        self.fc_K = nn.Linear(k_size, hid)
        self.fc_V = nn.Linear(v_size, hid)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc_s = nn.Linear(ori_size,hid)
        self.fc = nn.Linear(hid, hid)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid)

    def forward(self,ori, q,k,v):
        batch_size = q.size(0)
        Q = self.fc_Q(q)
        K = self.fc_K(k)
        V = self.fc_V(v)
        Q = Q.view(batch_size , -1, self.hid)
        K = K.view(batch_size , -1, self.hid)
        V = V.view(batch_size , -1, self.hid)

        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, self.hid)

        out = self.fc(context)
        out = self.dropout(out)
        ori = self.fc_s(ori)
        out = out + ori
        out = self.layer_norm(out)

        return out
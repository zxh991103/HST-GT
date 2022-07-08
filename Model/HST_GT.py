import torch
from Attention_Model import *
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers,temporal_features,back_features,data):
        super(HGT,self).__init__()
        self.hidden_size = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()

        embedding_size = 32

        for node_type in data.node_types:
            self.lin_dict[node_type] = nn.Embedding(512,embedding_size)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin_dict_out_hid = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict_out_hid[node_type] = Linear(8*hidden_channels, 8*hidden_channels)
        
        self.lin_dict_out = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict_out[node_type] = Linear(-1, 1)
        
        self.pack_fc1 = Linear(16*hidden_channels,16*hidden_channels)
        self.pack_fc2 = Linear(16*hidden_channels,1)

        self.lin_dict_out_hid['pack'] = self.pack_fc1
        self.lin_dict_out['pack'] = self.pack_fc2

        self.grus = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.grus[node_type] = nn.GRU(input_size=hidden_channels,hidden_size=hidden_channels, batch_first=True)
        
        self.gru_lin_in = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.gru_lin_in[node_type] = Linear(temporal_features,hidden_channels)

        self.back_lin = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.back_lin[node_type] = Linear(back_features,hidden_channels)

        self.gru_init_in = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.gru_init_in[node_type] = Linear(hidden_channels,hidden_channels)

        
        self.att_w = Attention(q_size=2 *hidden_channels,k_size=hidden_channels,v_size=hidden_channels,hid=hidden_channels)
        self.att_s = Attention(q_size= hidden_channels,k_size= 2*hidden_channels,v_size=2*hidden_channels,hid=hidden_channels)
        self.att_w_up = S_Attention(ori_size=hidden_channels, q_size= hidden_channels,k_size=2*hidden_channels,v_size=2*hidden_channels,hid=hidden_channels)
        self.att_s_up = S_Attention(ori_size=hidden_channels, q_size= 2*hidden_channels,k_size= hidden_channels,v_size=hidden_channels,hid=hidden_channels)

        self.fc_down = Linear(4*hidden_channels,4*hidden_channels)
        self.fc_context = Linear(4*hidden_channels,4*hidden_channels)

    def forward(self, x_dict, edge_index_dict,train_temporal_data,train_back_data,mask_in,mask_pack_in,if_first,cmask,dmask):
        out = {}
        for node_type, x in x_dict.items():
            out[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            out = conv(out, edge_index_dict)
        
        xs = out # X_s


        gru_in = {}
        for node_type,x in train_temporal_data.items():
            gru_in[node_type] = self.gru_lin_in[node_type](x).view(-1,1,self.hidden_size)
        

        back = {}
        for node_type,x in train_back_data.items():
            back[node_type] = self.back_lin[node_type](x)



        if if_first:

            gru_init = {}
            for node_type,x in out.items():
                gru_init[node_type] = self.gru_init_in[node_type](x).view(1,-1,self.hidden_size)

            
            gru_out = {}
            gru_hid = {}

            for i in gru_init:
                gru_out[i],gru_hid[i] = self.grus[i](gru_in[i],gru_init[i])
        else:
            gru_out = {}
            gru_hid = {}

            for i in gru_in:
                gru_out[i],gru_hid[i] = self.grus[i](gru_in[i])


        for i in gru_out:
            gru_out[i] = gru_out[i].view(-1,self.hidden_size)

        xt = {}

        for i in gru_out:
            xt[i] = (gru_out[i],back[i])
            xt[i] = torch.cat(xt[i],dim=1)
        
       
       # xs * \mul 512 xt * \mul 1024



        xst_w = self.att_w(xt['store'],xs['store'],xs['store'])
        xst_s = self.att_s(xs['sort'],xt['sort'],xt['sort'])

        xst = {
            'store':xst_w,
            'sort':xst_s
        }
        # xst * \mul 512


        xup_w = self.att_w_up(xs['store'],xs['store'],xt['store'],xt['store'])
        xup_s = self.att_s_up(xs['sort'],xt['sort'],xs['sort'],xs['sort'])

        xs['store'] = xup_w
        xs['sort'] = xup_s



        x_all = {
            'store':torch.cat((xs['store'],xt['store'],xst['store']),dim=1),
            'sort':torch.cat((xs['sort'],xt['sort'],xst['sort']),dim=1)

        }

        x_store = torch.matmul(mask_in['store'],x_all['store'])

        x_context = self.fc_context(x_all['sort'])
        x_down = self.fc_down(x_all['sort'])
        
        x_context = torch.matmul(cmask,x_context)
        x_down = torch.matmul(dmask,x_down)

        # cat x_store x_down
        # cat x_all['sort'] x_context


        x_out1 = torch.cat((x_store,x_down),dim=1)
        x_out3 = torch.cat((x_all['sort'],x_context),dim=1)

        x_out2 = torch.cat((x_out1,torch.matmul(mask_pack_in['sort'],x_out3)),dim=1)
        x_out3 = torch.matmul(mask_in['sort'],x_out3)
        
        x_out = {
            'store':x_out1,
            'pack':x_out2,
            'sort':x_out3
        }

        f_out = {}
        for i in x_out:
            f_out[i] = self.lin_dict_out_hid[i](x_out[i])
            f_out[i] = F.relu(f_out[i])
            f_out[i] = self.lin_dict_out[i](f_out[i])


        
        return f_out

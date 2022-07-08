SortN = 559
StoreN = 334

from datetime import datetime




def spatial_data(path):
    import json
    import os


    import numpy as np
    import pandas as pd
    import pandas as pd
    pd.set_option("display.max_rows",100)
    pd.set_option("display.max_columns",100)

    data = pd.read_csv(path)

    hj = [data['1'].tolist(),data['2'].tolist(),data['3'].tolist(),data['4'].tolist(),data['5'].tolist()]

    edges = {}
    L = len(data)
    for i in range(L):
        for j in range(4):
            num = hj[j][i]
            if  num != -1:
                if num not in edges:
                    edges[num] = []
                if hj[j+1][i] == -1:
                    continue
                else:
                    edges[num].append(hj[j+1][i])


    edgegraph = {}

    for i in edges:
        edgegraph[i] = {}
        for j in edges[i]:
            if j not in edgegraph[i]:
                edgegraph[i][j] = 0
            edgegraph[i][j] +=1


    edgegraphf = {}

    for i in edgegraph:
        if len(edgegraph[i]) == 0:
        
            continue
        else:
            edgegraphf[i] = edgegraph[i]

    SortN = 559 # 0<=i<=558
    StoreN = 334 # 0<=i<=333

    storeedge = {}
    stores = data['store'].tolist()
    for i,j in zip(stores,hj[0]):
        if i not in storeedge:
            storeedge[i] = {}
        if j not in storeedge[i]:
            storeedge[i][j] = 0
        storeedge[i][j] += 1

    import numpy as np



    N = 559
    S = np.zeros([N, N])
    for i in range(N):
        S[i][i] = 1
    for i in edgegraph:
        for j in edgegraph[i]:
            S[i][j] = edgegraph[i][j]
            
    for j in range(N):
        sum_of_col = sum(S[:,j])
        
        for i in range(N):
            S[i, j] /= sum_of_col

    alpha = 0.85
    A = alpha*S + (1-alpha) / N * np.ones([N, N])
    
    P_n = np.ones(N) / N
    P_n1 = np.zeros(N)

    e = 100000  # 误差初始化
    k = 0   # 记录迭代次数

    while e > 0.0000000001:   # 开始迭代
        P_n1 = np.dot(A, P_n)   # 迭代公式
        e = P_n1-P_n
        e = max(map(abs, e))    # 计算误差
        P_n = P_n1
        k += 1
        

    



    return SortN,StoreN,edgegraphf,storeedge,P_n





def date2index(s):
    basetime = datetime.strptime("2022-02-04 00:00:00","%Y-%m-%d %H:%M:%S")
    thisdate = datetime.strptime(s.replace(".0",""),"%Y-%m-%d %H:%M:%S")
    thisdate = datetime(
        year=thisdate.year,
        month=thisdate.month,
        day=thisdate.day,
        hour=thisdate.hour,
        minute=0,
        second=0
    )
    index = (thisdate-basetime).total_seconds()
    index = int(index/3600)
    return index


def str2date(s):
    return datetime.strptime(s.replace(".0",""),"%Y-%m-%d %H:%M:%S")

def get_key(l):
    t = []
    for i in l:
        if i != -1:
            t.append(str(i))
    return "_".join(t)


def get_train_spatial_temporal_data(path):
    _,_,_,_,P_n = spatial_data()

    allstd = 34.42220348436052
    allmean = 29.180005298727902

    whmean = 7.676320814410215
    packmean = 0.7230380924421469
    sortmean = 20.780646391539193

    


    import json
    import os
    import numpy as np
    import pandas as pd
    import pandas as pd
    pd.set_option("display.max_rows",100)
    pd.set_option("display.max_columns",100)
    data = pd.read_csv(path)

    train_data_wh = np.zeros((31*24,334,4))  # 0 仓库剩余处理量， 1 当前小时销量， 2 当日销量，3 下游线路通量和
    train_data_sort = np.zeros((31*24,559,4)) # 0 线路通量， 1 上游c 2 当前小时销量 3 当日销量

    train_data_background_wh = np.zeros((31*24,334,3)) # 0 日常/促销 1 是否周末 2 是否工作时间 7-21
    train_data_background_sort = np.zeros((31*24,559,3)) # 0 日常/促销 1 是否周末 2 是否工作时间 7-21


    y_all = [{} for i in range(31*24)]
   

    
    y_all_l = [{} for i in range(31*24)]
    
    for i in data.values:
        iwhin = str2date(i[0])
        iwhout = str2date(i[1])
        isortin = str2date(i[3])
        isortout = str2date(i[4])

        index = date2index(i[0])
        # print(index)
       
        t1 = (iwhout - iwhin).total_seconds()/3600
        t2 = (isortout - isortin).total_seconds()/3600
        pack = (isortin - iwhout).total_seconds()/3600
        tall = (isortout - iwhin).total_seconds()/3600

        if tall > 132:
            continue
        

        tall = (tall-allmean)/allstd
        t1 = (t1-whmean)/allstd
        pack =(pack-packmean)/allstd
        t2 = (t2-sortmean)/allstd

        key_all = get_key([i[2],i[5],i[6],i[7],i[8],i[9]])

       
        if key_all not in y_all_l[index]:
            y_all_l[index][key_all] = []
      
        y_all_l[index][key_all].append([t1,pack,t2,tall])
    
    for i in range(31*24):
        for j in y_all_l[i]:
            t = np.array(y_all_l[i][j])
            y_all[i][j] = np.mean(t,axis=0)

    wh_lists = []
    sort_lists = []
    y_all_t = []
    y_wh_t = []
    y_sort_t = []
    y_pack_t = []


    

    wh_masks = []
    sort_masks = []

    wh_pack_masks = []
    sort_pack_masks = []

    downstream_masks = []
    context_masks = []

    for i in y_all:
        wht,sortt,yallt,ywht,ysortt,ypackt = dict_to_labeldata(i)
        y_all_t.append(yallt)
        y_wh_t.append(ywht)
        y_sort_t.append(ysortt)
        y_pack_t.append(ypackt)


        wh_maskt = wh_list_to_mask(wht)
        sort_maskt = sort_list_to_mask(sortt)
        wh_pack_maskt = wh_maskt
        sort_pack_maskt = sort_list_to_packmask(sortt)


        # contextual mask
        cmask = contextual_mask(sortt,P_n)

        # downstream mask
        dmask = downstream_mask(wht,sortt,P_n)
        

        wh_masks.append(wh_maskt)
        sort_masks.append(sort_maskt)

        wh_pack_masks.append(wh_pack_maskt)
        sort_pack_masks.append(sort_pack_maskt)

        downstream_masks.append(dmask)
        context_masks.append(cmask)

    
        
    for i in data.values:
        if (str2date(i[4])-str2date(i[0])).total_seconds()/3600 > 132:
            continue


        iwhin = date2index(i[0])
        iwhout = date2index(i[1])
        for j in range(iwhin,iwhout+1):
            train_data_wh[j][i[2]][0] += 1 # wh 0 

            for sortnum in range(5,10):
                if i[sortnum] != -1:
                    train_data_sort[j][i[sortnum]][1] += 1 # sort 1

            
        train_data_wh[iwhin][i[2]][1] += 1 # wh 1
        
        for sortnum in range(5,10):
                if i[sortnum] != -1:
                    train_data_sort[iwhin][i[sortnum]][2] += 1 # sort 2

        

        isortin = date2index(i[3])
        isortout = date2index(i[4])

        for j in range(isortin,isortout+1):
            train_data_wh[j][i[2]][3] += 1 # wh 3
            for sortnum in range(5,10):
                train_data_sort[j][i[sortnum]][0] += 1 # sort 0


        dayend = (int(iwhin/24)+1)*24-1

        for j in range(iwhin,dayend+1):
            train_data_wh[j][i[2]][2] += 1 # wh 2 

            for sortnum in range(5,10):
                if i[sortnum] != -1:
                    train_data_sort[j][i[sortnum]][3] += 1 # sort 3
        
    from datetime import datetime,timedelta
    basetime = datetime.strptime("2022-02-04 00:00:00","%Y-%m-%d %H:%M:%S")
    for i in range(31*24):
        t = basetime + timedelta(hours=i)
        for j in range(334):
            if t.month == 3 and t.day >= 2:
                train_data_background_wh[i][j][0] = 1 # bg 0
            if t.weekday() > 4 :
                train_data_background_wh[i][j][1] = 1 # bg 1
            if t.hour >= 7 and t.hour <= 21:
                train_data_background_wh[i][j][2] = 1 # bg 2
        for j in range(559):
            if t.month == 3 and t.day >= 2:
                train_data_background_sort[i][j][0] = 1 # bg 0
            if t.weekday() > 4 :
                train_data_background_sort[i][j][1] = 1 # bg 1
            if t.hour >= 7 and t.hour <= 21:
                train_data_background_sort[i][j][2] = 1 # bg 2

    return  train_data_wh,train_data_sort ,train_data_background_wh,train_data_background_sort,y_all_t \
            ,y_wh_t ,y_sort_t ,y_pack_t ,wh_masks,sort_masks,wh_pack_masks,sort_pack_masks,downstream_masks,context_masks


def contextual_mask(s_list,P_n):
    cmask = np.zeros((559,559))
    for i in s_list:
        n = len(i)
        for j in range(n):
            t = []
            for k in range(n):
                if j != k:
                    t.append(P_n[i[k]]*(np.e**((3-np.abs(j-k))/10)))
            t = np.array(t)
            t = t/np.sum(t)
            cnt = 0
            for k in range(n):
                if j!=k:
                    cmask[i[j]][i[k]] = t[cnt]
                    cnt+=1
    return cmask
                    

def downstream_mask(w_list,s_list,P_n):
    N = len(w_list)
    dmask = np.zeros((N,559))
    for i in range(N):
        t = []
        s = 0
        for j in s_list[i]:
            t.append(P_n[j]* (np.e**((4-s)/10)) )
            s += 1
        t = np.array(t)
        t = t/np.sum(t)
        
        for j,score in zip(s_list[i],t):
            dmask[i][j] = score
    return dmask





import numpy as np
def wh_list_to_mask(wh_lists):
    N = len(wh_lists)
    wh_mask = np.zeros((N,334))
    for i in range(N):
        wh_mask[i][wh_lists[i][0]] = 1
    return torch.from_numpy(wh_mask)

def sort_list_to_mask(sort_lists):
    N = len(sort_lists)
    sort_mask = np.zeros((N,559))
    for i in range(N):
        for j in sort_lists[i]:
            sort_mask[i][j] = 1
    return torch.from_numpy(sort_mask)

def sort_list_to_packmask(sort_lists):
    N = len(sort_lists)
    sort_mask = np.zeros((N,559))
    for i in range(N):
        sort_mask[i][sort_lists[i][0]] = 1
    return torch.from_numpy(sort_mask)

def dict_to_labeldata(y_all):
    wh_lists = []
    sort_lists = []
    y_all_t = []
    y_sort_t = []
    y_wh_t = []
    y_pack_t = []
    for i in y_all:
        t = y_all[i]
        l = i.split("_")
        wh_list = [int(l[0])]
        sort_list = []
        for j in range(1,len(l)):
            sort_list.append(int(l[j]))
        
        wh_lists.append(wh_list)
        sort_lists.append(sort_list)

        y_all_t.append(t[3])
        y_wh_t.append(t[0])
        y_sort_t.append(t[2])
        y_pack_t.append(t[1])
    
    return wh_lists,sort_lists,y_all_t,y_wh_t,y_sort_t,y_pack_t





def get_spatial_data(path):
    import json
    import os
    import numpy as np
    import pandas as pd
    import pandas as pd
    pd.set_option("display.max_rows",100)
    pd.set_option("display.max_columns",100)
    data = pd.read_csv(path)


    y_all = {}
    
    y_all_l = {}
    
    for i in data.values:
        iwhin = str2date(i[0])
        iwhout = str2date(i[1])
        isortin = str2date(i[3])
        isortout = str2date(i[4])
        index = date2index(i[0])
        # print(index)
        t1 = (iwhout - iwhin).total_seconds()
        t2 = (isortout - isortin).total_seconds()
        pack = (isortin - iwhout).total_seconds()
        tall = (isortout - iwhin).total_seconds()
        key_all = get_key([i[2],i[5],i[6],i[7],i[8],i[9]])

        if key_all not in y_all_l:
            y_all_l[key_all] = []
        y_all_l[key_all].append([t1,pack,t2,tall])
    
    for j in y_all_l:
        t = np.array(y_all_l[j])
        y_all[j] = np.mean(t,axis=0)
    SortN,StoreN,edgegraphf,storeedge,_ = spatial_data()
    from sklearn import preprocessing  
    store = [[i] for i in range(StoreN)]
    sort =[[i] for i in range(SortN)]
    enc1 = preprocessing.OneHotEncoder()
    enc1.fit(store)
    store = enc1.transform(store).toarray()
    enc2 = preprocessing.OneHotEncoder()
    enc2.fit(sort)
    sort = enc2.transform(sort).toarray()
    edge_sort = []
    for i in edgegraphf:
        t = []
        for j in edgegraphf[i]:
            t = [i,j]
            edge_sort.append(t)
            t = [j,i]
            edge_sort.append(t)
    edge_sort = np.array(edge_sort).T
    edge_store_sort = []
    for i in storeedge:
        t = []
        for j in storeedge[i]:
            t = [i,j]
            edge_store_sort.append(t)

    edge_sort_store = []
    for i in storeedge:
        t = []
        for j in storeedge[i]:
            t = [j,i]
            edge_sort_store.append(t)
    edge_store_sort = np.array(edge_store_sort).T
    edge_sort_store = np.array(edge_sort_store).T
    wh_lists,sort_lists,y_all_t,y_wh_t,y_sort_t,y_pack_t = dict_to_labeldata(y_all)
    wh_mask = wh_list_to_mask(wh_lists)
    sort_mask = sort_list_to_mask(sort_lists)
    wh_pack_mask = wh_mask
    sort_pack_mask = sort_list_to_packmask(sort_lists)
    return store,sort,edge_store_sort,edge_sort_store,edge_sort,\
        wh_mask,sort_mask,wh_pack_mask,sort_pack_mask,y_all_t,y_wh_t,y_sort_t,y_pack_t


# Graph Data
import torch
class graph_data:
    def __init__(self,store,sort,edge_store_sort,edge_sort_store,edge_sort,device = 'cpu'):
        self.node_types = ['store','sort']
        self.edge_types = [('store','to','sort'),('sort','to','store'),('sort','to','sort')]
        self.meta_data = (self.node_types,self.edge_types)
        self.x_dict = {}
        self.x_dict['store'] = torch.from_numpy(store).to(torch.long).view(334,-1).to(device)
        self.x_dict['sort'] = torch.from_numpy(sort).to(torch.long).view(559,-1).to(device)

        self.edge_index_dict = {}
        self.edge_index_dict[('store','to','sort')] = torch.from_numpy(edge_store_sort).to(torch.long).view(2,-1).to(device)
        self.edge_index_dict[('sort','to','store')] = torch.from_numpy(edge_sort_store).to(torch.long).view(2,-1).to(device)
        self.edge_index_dict[('sort','to','sort')] = torch.from_numpy(edge_sort).to(torch.long).view(2,-1).to(device)
    def metadata(self):
        return self.meta_data





def spatial_graph_data():
    
    store,sort,edge_store_sort,edge_sort_store,edge_sort,\
        _,_,_,_,_,_,_,_ = get_spatial_data(path="train.csv")
    data = graph_data(store,sort,edge_store_sort,edge_sort_store,edge_sort,'cuda:0')
    return data


def temporal_data(path):
    train_data_wh,train_data_sort ,train_data_background_wh,train_data_background_sort,y_all_t \
    ,y_wh_t ,y_sort_t ,y_pack_t ,wh_masks,sort_masks,wh_pack_masks,sort_pack_masks,downstream_masks,context_masks = get_train_spatial_temporal_data(path)
    T = len(wh_masks)
    mask_ins = []
    mask_pack_ins = []
    downmask_ins = []
    cmask_ins = []

    y_all_ts = [] 
    y_wh_ts = []
    y_sort_ts = []
    y_pack_ts = []

    train_data_wh = torch.from_numpy(train_data_wh).to('cuda:0').to(torch.float32)
    train_data_sort = torch.from_numpy(train_data_sort).to('cuda:0').to(torch.float32)
    train_data_background_wh = torch.from_numpy(train_data_background_wh).to('cuda:0').to(torch.float32)
    train_data_background_sort = torch.from_numpy(train_data_background_sort).to('cuda:0').to(torch.float32)

    for i in range(T):
        wh_mask = wh_masks[i].to("cuda:0").to(torch.float32)
        sort_mask = sort_masks[i].to('cuda:0').to(torch.float32)

        mask_in = {
            'store':wh_mask,
            'sort':sort_mask
        }
        wh_pack_mask = wh_pack_masks[i].to("cuda:0").to(torch.float32)
        sort_pack_mask = sort_pack_masks[i].to('cuda:0').to(torch.float32)
        mask_pack_in = {
            'store':wh_pack_mask,
            'sort':sort_pack_mask
        }

        mask_ins.append(mask_in)
        mask_pack_ins.append(mask_pack_in)
        
        y_all_ts.append(torch.from_numpy(np.array(y_all_t[i])).to('cuda:0').to(torch.float32).view(-1,1))
        y_wh_ts.append(torch.from_numpy(np.array(y_wh_t[i])).to('cuda:0').to(torch.float32).view(-1,1))
        y_sort_ts.append(torch.from_numpy(np.array(y_sort_t[i])).to('cuda:0').to(torch.float32).view(-1,1))
        y_pack_ts.append(torch.from_numpy(np.array(y_pack_t[i])).to('cuda:0').to(torch.float32).view(-1,1))

        downmask_ins.append(torch.from_numpy(downstream_masks[i]).to('cuda:0').to(torch.float32))
        cmask_ins.append(torch.from_numpy(context_masks[i]).to('cuda:0').to(torch.float32))
        
    return train_data_wh,train_data_sort ,train_data_background_wh,train_data_background_sort,\
        mask_ins,mask_pack_ins,y_all_ts,y_wh_ts,y_sort_ts,y_pack_ts,downmask_ins,cmask_ins


allstd = 34.42220348436052

allmean = 29.180005298727902
whmean = 7.676320814410215
packmean = 0.7230380924421469
sortmean = 20.780646391539193
from Model.HST_GT import *
from Utils.Data_utils import *

data = spatial_graph_data()

train_data_wh,train_data_sort ,train_data_background_wh,train_data_background_sort,\
        mask_ins,mask_pack_ins,y_all_ts,y_wh_ts,y_sort_ts,y_pack_ts,downmask_ins,cmask_ins = temporal_data("Data/train.csv")

test_data_wh,test_data_sort ,test_data_background_wh,test_data_background_sort,\
        test_mask_ins,test_mask_pack_ins,test_y_all_ts,test_y_wh_ts,test_y_sort_ts,test_y_pack_ts,test_downmask_ins,test_cmask_ins = temporal_data("Data/test.csv")


model = HGT(hidden_channels=64,  num_heads=4, num_layers=3,temporal_features=4,back_features=3,data=data)
device = 'cuda:0'


if __name__ == "__main__":
    T_train = len(mask_ins)
    T_test = len(test_cmask_ins)
    beta = 0.8
    epoch_num = 100000
    model.train()
    model.cuda()
    model.float()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00005)
    
    lossfunction = torch.nn.MSELoss()
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    starttime = datetime.datetime.now()

    if_first = True
    for epoch in range(epoch_num):
        

        loss_whs = 0
        loss_packs =0
        loss_sorts = 0
        loss_alls = 0
        losss = 0
        sn = 0
        model.train()
        for t in range(T_train):
            train_temporal_data = {
                'store':train_data_wh[t].view(-1,4),
                'sort':train_data_sort[t].view(-1,4)
            }
            train_back_data = {
                'store':train_data_background_wh[t].view(-1,3),
                'sort':train_data_background_sort[t].view(-1,3)
            }
            mask_in = mask_ins[t]
            mask_pack_in = mask_pack_ins[t]
            y_wh_t = y_wh_ts[t]
            y_pack_t = y_pack_ts[t]
            y_sort_t = y_sort_ts[t]
            y_all_t = y_all_ts[t]

            cmask = cmask_ins[t]
            dmask = downmask_ins[t]
            out = model(x_dict, edge_index_dict,train_temporal_data,train_back_data,mask_in,mask_pack_in,if_first,cmask,dmask)
            if_first = False
            

            loss_wh = lossfunction(out['store'],y_wh_t)
            loss_sort = lossfunction(out['sort'],y_sort_t)
            loss_pack = lossfunction(out['pack'],y_pack_t)
            loss_all = lossfunction(out['store']+out['sort']+out['pack'],y_all_t)
            loss = loss_wh + loss_sort + beta * loss_all + loss_pack
            optimizer.zero_grad()
            if len(y_all_t) == 0:
                continue
            sn += len(y_all_t)
            tn = len(y_all_t)

            loss_whs += float(loss_wh) * tn
            loss_packs += float(loss_pack) * tn
            loss_sorts += float(loss_sort) * tn
            loss_alls += float(loss_all) * tn
            losss += float(loss) * tn
            
            loss.backward()
            optimizer.step()
            if epoch % 2 == 0:

                with open("train_log.txt",'a+') as f:
                    print("{} {} {} {} {} {} {}".format(epoch,t,float(loss_wh),float(loss_pack),float(loss_sort),float(loss_all),float(loss)),
                    file=f)

        if epoch % 2 == 0:
            endtime = datetime.datetime.now()
            torch.save(model,"Model_Save/epoch_{}.pkl".format(epoch))
            with open("train.txt",'a+') as f:
                print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn),file=f)
            print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}, time: {}s".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn,(endtime-starttime).seconds))
            starttime = datetime.datetime.now()
        
        # test
        if epoch % 100 == 0:
            model.eval()
            loss_whs = 0
            loss_packs =0
            loss_sorts = 0
            loss_alls = 0
            losss = 0
            sn = 0

            for t in range(T_train):
                train_temporal_data = {
                    'store':train_data_wh[t].view(-1,4),
                    'sort':train_data_sort[t].view(-1,4)
                }
                train_back_data = {
                    'store':train_data_background_wh[t].view(-1,3),
                    'sort':train_data_background_sort[t].view(-1,3)
                }
                mask_in = mask_ins[t]
                mask_pack_in = mask_pack_ins[t]
                y_wh_t = y_wh_ts[t]
                y_pack_t = y_pack_ts[t]
                y_sort_t = y_sort_ts[t]
                y_all_t = y_all_ts[t]

                cmask = cmask_ins[t]
                dmask = downmask_ins[t]
                out = model(x_dict, edge_index_dict,train_temporal_data,train_back_data,mask_in,mask_pack_in,if_first,cmask,dmask)                

                loss_wh = lossfunction(out['store'],y_wh_t)
                loss_sort = lossfunction(out['sort'],y_sort_t)
                loss_pack = lossfunction(out['pack'],y_pack_t)
                loss_all = lossfunction(out['store']+out['sort']+out['pack'],y_all_t)
                loss = loss_wh + loss_sort + beta * loss_all + loss_pack
                
                if len(y_all_t) == 0:
                    continue
                sn += len(y_all_t)
                tn = len(y_all_t)

                loss_whs += float(loss_wh) * tn
                loss_packs += float(loss_pack) * tn
                loss_sorts += float(loss_sort) * tn
                loss_alls += float(loss_all) * tn
                losss += float(loss) * tn
                
            
                if epoch % 2 == 0:

                    with open("test_log.txt",'a+') as f:
                        print("{} {} {} {} {} {} {}".format(epoch,t,float(loss_wh),float(loss_pack),float(loss_sort),float(loss_all),float(loss)),
                        file=f)

            if epoch % 2 == 0:
                endtime = datetime.datetime.now()
                with open("test.txt",'a+') as f:
                    print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn),file=f)
                print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}, time: {}s".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn,(endtime-starttime).seconds))
                starttime = datetime.datetime.now()
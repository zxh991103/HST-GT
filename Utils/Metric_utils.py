from sklearn.metrics import r2_score
def r2(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = r2_score(pre,yt)
    return sc

from sklearn.metrics import mean_squared_error, mean_absolute_error
def mse(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = mean_squared_error(pre,yt)
    return sc

def mae(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = mean_absolute_error(pre,yt)
    return sc

def rmse(pre,yt):
    sc = mse(pre,yt) ** 0.5
    return sc


import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def mape(pre,yt):
    y_pred = pre.view(-1).cpu().detach().numpy()
    y_true = yt.view(-1).cpu().detach().numpy()

    return mean_absolute_percentage_error(pre,yt)

def smape(pre,yt):
    y_pred = pre.view(-1).cpu().detach().numpy()
    y_true = yt.view(-1).cpu().detach().numpy()
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def metric(pre,yt):
    res = {
        "r2_score":r2(pre,yt),
        "mae":mae(pre,yt),
        "mape":mape(pre,yt),
        "smape":smape(pre,yt),
        "mse":mse(pre,yt),
        "rmse":rmse(pre,yt)
    }
    return res

def metric_real(pre,yt,nm):
    pass
    allstd = 34.42220348436052
    allmean = 29.180005298727902
    whmean = 7.676320814410215
    packmean = 0.7230380924421469
    sortmean = 20.780646391539193
    st_d = {
        "all":allmean,
        "store":whmean,
        "pack":packmean,
        "sort":sortmean
    }
    pre = (pre * allstd) + st_d[nm]
    yt = (yt * allstd) + st_d[nm]

    return metric(pre,yt)

def metric_all(pres,yts):
    res = {
        "norm":{},
        "real":{}
    }
    for i in pres:
        res["norm"][i] = metric(pres[i],yts[i])
        res["real"][i] = metric_real(pres[i],yts[i],i)
    return res


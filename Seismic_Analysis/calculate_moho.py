import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

EvtTime = datetime.time(17, 47, 16, 0)
PSphase = {'P':'T1', 'S':'T2'}

# 讀取測站數據
df_sta = pd.read_csv('ChiChi/wilber-stations.txt', sep='|')
# 震中距單位轉換(度轉公里)
df_sta.Distance = df_sta.Distance * 111.1

# 讀取事件數據
with open ('ChiChi/phase.txt') as fg:
    lines = fg.readlines()

df_evt = pd.DataFrame()
for line in lines:
    try:
        phase = line.rstrip().split()[3]
    except:
        continue

    area, sta, ang, phase, date, h, m, ss, amp, M, B = line.rstrip().split()
    s, ms = ss.split('.')
    h = int(h)
    m = int(m)
    s = int(s)
    ms = int(ms)
    t1 = parser.parse(str(datetime.time(h, m, s, ms)))
    t0 = parser.parse(str(EvtTime))
    TravelTime = (t1 - t0).seconds
    df_evt = df_evt.append({'Sta': sta, 'PhaseInfo': phase, 'Date': date, 'TravelTime': TravelTime}, ignore_index=True)

Ok = []
for phase in PSphase.values():
    dff = df_evt[df_evt['PhaseInfo'].str.endswith(phase)].copy()
    dff = dff.reset_index(drop=True)
    for i in range(len(dff)):
        sta = dff.iloc[i].Sta
        dff.loc[i, 'Distance'] = df_sta[df_sta.Station == sta].Distance.item()
        dff = dff.sort_values('Distance')
    Ok.append(dff)

# T-Distance
Vel = []
CrossIndexs = [5, 5]

for n, key in enumerate(PSphase.keys()):
    # 線性擬合
    df = Ok[n]
    CrossIndex = CrossIndexs[n]
    XgGroup = df.iloc[:CrossIndex + 1]
    XnGroup = df.iloc[CrossIndex:]

    pg = np.polyfit(list(XgGroup.Distance), list(XgGroup.TravelTime), 1)
    fg = np.poly1d(pg)
    pn = np.polyfit(list(XnGroup.Distance), list(XnGroup.TravelTime), 1)
    fn = np.poly1d(pn)
    Vg = np.round(1 / pg[0], 3)
    Vn = np.round(1 / pn[0], 3)
    Vel.append([Vg, Vn])

    # 誤差分析
    R2_g = round(r2_score(list(XgGroup.TravelTime), fg(list(XgGroup.Distance))), 3)
    R2_n = round(r2_score(list(XnGroup.TravelTime), fn(list(XnGroup.Distance))), 3)
    RMSE_g = round(np.sqrt(mean_squared_error(list(XgGroup.TravelTime), fg(list(XgGroup.Distance)))), 3)
    RMSE_n = round(np.sqrt(mean_squared_error(list(XnGroup.TravelTime),
                                        fn(list(XnGroup.Distance)))), 3)
    print('R2_g', R2_g, '\nR2_n', R2_n)
    print('RMSE_g', RMSE_g, '\nRMSE_n', RMSE_n)
    text1 = f'{key}g: {Vg} km/s\nR2 score: {R2_g}\nRMSE: {RMSE_g}\n\n{key}n: {Vn} km/s\nR2 score: {R2_n}\nRMSE: {RMSE_n}\n\n'
    print(text1)

    # 找到臨界震中距
    # x = np.linspace(min(df.Distance), max(df.Distance), 100)
    Xg_Ypred = fg(list(df.Distance))
    Xn_Ypred = fn(list(XnGroup.Distance))

    for i in np.linspace(min(df.Distance), max(df.Distance), 100):
        if fg(i) > fn(i):
            Xcross = round(i, 0)
            Ycross = fg(i)
            break
    text2 = f'Xc: {Xcross}km'
    print(text2)

    # 計算莫赫面深度
    if key == 'P':
        moho_H = Xcross * (((Vn-Vg)/(Vn+Vg))**(1/2)) / 2
        moho_H = np.round(moho_H, 3)
        text3 = f'Depth of Moho: {moho_H} km'
        print(text3)

    print('\n')
    # 畫圖檢測
    plt.figure()
    # 標註測站
    for i in range(len(df)):
        plt.annotate(df.iloc[i].Sta, [df.iloc[i].Distance, df.iloc[i].TravelTime], xytext=(-20,30), textcoords="offset pixels", fontsize=6)
    plt.scatter(df.Distance, df.TravelTime)

    plt.plot(list(df.Distance), Xg_Ypred, 'b', label=f'{key}g wave')
    plt.plot(list(XnGroup.Distance), Xn_Ypred, 'r', label=f'{key}n wave')
    plt.title(f'Travel time ({key}) - Distance for Chi-Chi eq')
    plt.xlabel('Epicenter Distance (Km)')
    plt.ylabel(f'T{key.lower()} (s)')
    # plt.xlim(0, 1000)
    # plt.ylim(0, 300)
    plt.text(50, 150, text1)
    if key == 'P':
        plt.text(1000, 50, text2+'\n'+text3)
        plt.plot((Xcross, Xcross), (0, Ycross), color='k', linestyle='--')
        plt.text(Xcross, -30, 'Xc', ha='center')

    plt.grid()
    plt.legend()
    plt.savefig(f'{key}.jpg')

# 計算泊松比
print('Passion ratio for crust:', round(Vel[0][0]/Vel[1][0], 3))
print('Passion ratio for upper mantle:', round(Vel[0][1]/Vel[1][1], 3))
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
import os

# 設定輸出檔
outpath = 'out.txt'
f = open(outpath, 'w', encoding='UTF-8', errors='replace')
col = ['序號','年','月','日', '時','分','秒','緯度','經度','深度','殘差均方根(s)']
col = ('\t').join(col)
f.write(col+'\n')
if not os.path.exists('Figure'):
    os.makedirs('Figure')

# 讀取測站數據
df_sta = pd.read_csv('station.txt', sep='\t')
la_str = [str(s) for s in df_sta.La.values]
la = [float(s[:2]) + float(s[2:4])/60 + float(s[4:6])/3600 for s in la_str]
lo_str = [str(s) for s in df_sta.Lo.values]
lo = [float(s[:3]) + float(s[3:5])/60 + float(s[5:7])/3600 for s in lo_str]

la_c = (max(la) + min(la)) / 2
lo_c = (max(lo) + min(lo)) / 2


# 檢查數據，再將經緯度轉直角座標系，並以台網中心為原點
STY = []
STX = []
for i in range(len(la)):
    if abs(la[i]) > 90:
        print(f'Error:第{i}筆數據緯度超出範圍')
    if abs(lo[i]) > 180:
        print(f'Error:第{i}筆數據經度超出範圍')
    sty = 111.199 * (la[i] - la_c)
    stx = 111.199 * (lo[i] - lo_c) * np.cos((la[i] + la_c) / 2)
    STY.append(sty)
    STX.append(stx)
df_sta.insert(4, 'X', STX)
df_sta.insert(5, 'Y', STY)

# 讀取事件數據
df_evts = pd.read_csv('December.txt', sep='\t')
evt_index0 = df_evts.loc[df_evts['XH'].notnull()].index
df_time = pd.read_csv('In.txt', sep='\t')

for i in range(len(evt_index0)):
    XH = df_evts['XH'].loc[df_evts['XH'].notnull()].values[i]
    try:
        df_evt = df_evts[evt_index0[i]:evt_index0[i + 1]].copy()

    except:
        df_evt = df_evts[evt_index0[i]:].copy()
    df_evt = df_evt.reset_index(drop=True)
    df_evt = df_evt.fillna(method='pad')
    single_evt = df_time[df_time['序號'] == XH]
    t0 = (single_evt['時']*60*60 + single_evt['分']*60 + single_evt['秒']).values

    # 調整震相到時欄位，以發震時間為0s
    ArrivalTimes = list(df_evt.DSH)
    TT = []
    X = []
    Y = []
    TM = []
    for ArrivalTime in ArrivalTimes:
        h, m, s = [int(i) for i in ArrivalTime.split('-')]
        T = (h * 60 * 60) + (m * 60) + (s / 10)
        TT.append(T)
    Time = TT - (t0 * np.ones(len(TT)))
    df_evt.insert(5, 'Time', Time)
    df_evt.insert(6, 'X', np.zeros(len(df_evt)))
    df_evt.insert(7, 'Y', np.zeros(len(df_evt)))

    # 連接兩表
    for i in range(len(df_evt)):
        sta = df_evt.iloc[i].TM
        df_evt.loc[i, 'X'] = df_sta[df_sta.Sta == sta].X.item()
        df_evt.loc[i, 'Y'] = df_sta[df_sta.Sta == sta].Y.item()

    # 只選用PG
    df_cleand = df_evt[df_evt['ZHX'] == 'PG']

    # 設定初始值
    x, y = [np.mean(df_cleand.X), np.mean(df_cleand.Y)]
    counter = 1
    Key = []

    for z in range(1, 40):
        min_r = 100
        # 設定不同層的P波波速
        if 0 < z <= 10:
            Vp = 4.5
        elif 10 < z <= 30:
            Vp = 6.0
        elif 30 < z <= 40:
            Vp = 7.5

        for counter in range(1, 30):

            xx = x * np.ones(len(df_cleand.X))
            yy = y * np.ones(len(df_cleand.X))
            zz = z * np.ones(len(df_cleand.X))
            t_pre = (((xx - df_cleand.X) ** 2 + (yy - df_cleand.Y) ** 2 + zz ** 2) ** (1 / 2)) / Vp
            res = df_cleand.Time - t_pre
            res_rms = ((sum(res**2))/len(res))**(1/2)

            Tx = (xx - df_cleand.X) * (((xx - df_cleand.X) ** 2 + (yy - df_cleand.Y) ** 2 + zz ** 2) ** (-1 / 2)) / Vp
            Ty = (yy - df_cleand.Y) * (((xx - df_cleand.X) ** 2 + (yy - df_cleand.Y) ** 2 + zz ** 2) ** (-1 / 2)) / Vp
            Tz = zz * (((xx - df_cleand.X) ** 2 + (yy - df_cleand.Y) ** 2 + zz ** 2) ** (-1 / 2)) / Vp
            G = np.array([Tx, Ty, Tz]).T

            left = inv(np.matmul(G.T, G))
            delta_m = np.matmul(np.matmul(left, G.T), res)

            if min_r > res_rms:
                source_la = (y / 111.199) + la_c
                source_lo = ((x / 111.199) / np.cos((source_la + la_c) / 2)) + lo_c
                source = [np.round(source_la, 4), np.round(source_lo, 4), np.round(z, 4), np.round(res_rms, 4)]
                min_r = res_rms

            x += delta_m[0]
            y += delta_m[1]

        Key.append(source)

    dff = pd.DataFrame(Key, columns=['La', 'Lo', 'Depth', 'Residual'])
    final_source = dff.sort_values('Residual').iloc[0]
    print(f'event {XH}\n', final_source)

    # 畫圖檢測
    Y = 111.199 * (final_source.La - la_c)
    X = 111.199 * (final_source.Lo - lo_c) * np.cos((final_source.La + la_c) / 2)
    plt.figure()
    c = plt.scatter(df_cleand.X, df_cleand.Y, marker='^', s=100, c=df_cleand.Time, cmap='bone')
    sta_txt = list(map(int, df_cleand.TM))
    for i, txt in enumerate(sta_txt):
        plt.annotate(txt, [df_cleand['X'].iloc[i], df_cleand['Y'].iloc[i]])
    plt.colorbar(c, label='Tpg (s)')
    plt.scatter(X, Y, marker='*', s=200, color='r')
    plt.title(f'event {XH} \nLa {final_source.La}  Lo {final_source.Lo}  Depth {final_source.Depth}km Residual {final_source.Residual}', fontsize=14)
    plt.savefig(f'Figure\event_{XH}.jpg')

    timetxt = single_evt.to_string(header=False, index=False).split()
    timetxt = ('\t').join(timetxt)
    loctxt = final_source.to_string(header=False, index=False).split()
    loctxt = ('\t').join(loctxt)
    f.write(timetxt+'\t'+loctxt+'\n')
f.close()
plt.show()
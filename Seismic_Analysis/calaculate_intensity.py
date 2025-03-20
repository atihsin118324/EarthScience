""" ZNE三分量加速度 計算新震度"""
import numpy as np
from scipy import signal
import glob
import pandas as pd
import scipy.integrate as spi


def get_acceleration_data_ZNE(event_path, start_line=11, station_line=0):
    time, Z, N, E = [], [], [], []

    with open(event_path) as f:
        line = f.readlines()
        station = line[station_line].split(':')[1].strip()

    for l in line[start_line:]:
        l.strip()
        time.append(l.split()[0])
        Z.append(float(l.split()[1]))
        N.append(float(l.split()[2]))
        E.append(float(l.split()[3]))
    return station, time, Z, N, E


def lowpass_filter(data=None, lf=10, n=4, nf=200):
    """
    data: 原始波型數據
    n:階數
    lf :截止頻率
    nf :取樣頻率
    (Wn : 2*lf/nf)
    """

    Wn = 2 * lf / nf
    b, a = signal.butter(n, Wn, 'lowpass')
    filtered = signal.filtfilt(b, a, data)
    return filtered


def calaulate_PGX(Z=None, N=None, E=None):
    ZNE = np.sqrt(Z ** 2 + N ** 2 + E ** 2)
    PGX = max(ZNE)
    return PGX


def PGA2intensity(PGA_group=None):
    intensity_group = []
    for PGA in PGA_group:
        if PGA < 0.8:
            intensity = 0
        elif 0.8 <= PGA < 2.5:
            intensity = 1
        elif 2.5 <= PGA < 8.0:
            intensity = 2
        elif 8.0 <= PGA < 25:
            intensity = 3
        elif 25 <= PGA < 80:
            intensity = 4
        elif 80 <= PGA < 140:
            intensity = 5
        elif 140 <= PGA < 250:
            intensity = 5.5
        elif 250 <= PGA < 440:
            intensity = 6.0
        elif 440 <= PGA < 800:
            intensity = 6.5
        else:
            intensity = 7
        intensity_group.append(intensity)
    return intensity_group


def integrate_acceleration(a, dt, v0=0.0, d0=0.0):
    v = spi.cumtrapz(a, dx=dt, initial=v0)
    return v


def PGV2intensity(PGV_group=None):
    intensity_group = []
    for PGV in PGV_group:
        if PGV < 0.2:
            intensity = 0
        elif 0.2 <= PGV < 0.7:
            intensity = 1
        elif 0.7 <= PGV < 1.9:
            intensity = 2
        elif 1.9 <= PGV < 5.7:
            intensity = 3
        elif 5.7 <= PGV < 15:
            intensity = 4
        elif 15 <= PGV < 30:
            intensity = 5
        elif 30 <= PGV < 50:
            intensity = 5.5
        elif 50 <= PGV < 80:
            intensity = 6.0
        elif 80 <= PGV < 140:
            intensity = 6.5
        else:
            intensity = 7
        intensity_group.append(intensity)
    return intensity_group


if __name__ == '__main__':
    StationGroup, PGAGroup, PathGroup, PGVGroup = [], [], [], []
    events = glob.glob('.\\20131031\\FreeField\\Record\\*.CVA.txt')
    for event in events:
        sta, time, Z, N, E = get_acceleration_data_ZNE(event, start_line=11,
                                                     station_line=0)
        Z_filtered = lowpass_filter(Z)
        N_filtered = lowpass_filter(N)
        E_filtered = lowpass_filter(E)
        PGA = calaulate_PGX(Z_filtered, N_filtered, E_filtered)

        PGAGroup.append(PGA)
        StationGroup.append(sta)
        PathGroup.append(event)

    d = {'Station': StationGroup, 'Path': PathGroup, 'PGA': PGAGroup}
    df = pd.DataFrame(d)
    df = df.sort_values(by=['PGA'], ascending=False)
    df['Intensity'] = PGA2intensity(df['PGA'])
    BiggerThen5 = df[df['Intensity'] >= 5].copy()
    print('\n ########### PGA and Intensity >= 5 ###########\n')
    print(BiggerThen5[['Station', 'PGA', 'Intensity']])

    for event in BiggerThen5['Path']:
        sta, time, Z, N, E = get_acceleration_data_ZNE(event)
        Z_vel = integrate_acceleration(Z, 0.05, v0=0.0, d0=0.0)
        N_vel = integrate_acceleration(N, 0.05, v0=0.0, d0=0.0)
        E_vel = integrate_acceleration(E, 0.05, v0=0.0, d0=0.0)
        Z_vel_filtered = lowpass_filter(Z_vel, lf=0.075)
        N_vel_filtered = lowpass_filter(N_vel, lf=0.075)
        E_vel_filtered = lowpass_filter(E_vel, lf=0.075)
        PGV = calaulate_PGX(Z_vel_filtered, N_vel_filtered, E_vel_filtered)
        PGVGroup.append(PGV)

    BiggerThen5.loc[:, 'PGV'] = PGVGroup
    Intensity_new = PGV2intensity(PGVGroup)
    Intensity_new = ([4 if I <= 4 else I for I in Intensity_new])
    BiggerThen5.loc[:, 'Intensity_new'] = Intensity_new

    print('\n ########### PGV and New Intensity ###########\n')
    print(BiggerThen5[['Station', 'PGV', 'Intensity_new']])
    print('\n\n')

    print('\n ########### final sorted intensity ###########\n')
    BiggerThen5 = BiggerThen5.sort_values(by=['Intensity_new'], ascending=False)
    print(BiggerThen5[['Station', 'PGV', 'Intensity_new']])
    print('\n\n')

    # BiggerThen5.to_excel("output.xlsx")
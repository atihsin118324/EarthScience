# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 04:31:32 2021

@author: at

1.加入rms(R)/rms(T) ratio , 高區設門檻
2.算 Rayleigh-wave H/V ratio 用均方根方法算
瞬時相位差
"""
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
import obspy
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy.signal import hilbert
import os
import pandas as pd

class HVRatioCalculator:
    def __init__(self, base_path, hv_file):
        self.base_path = base_path
        self.hv10 = pd.read_table(hv_file, sep='\s+', header=None)
        self.pwd = os.getcwd()
        self.all_event = os.listdir(os.path.join(base_path, 'ori_waveform'))

    def process_event(self, event):
        print(f"It's working on event {event}")
        self.create_directory(os.path.join(self.base_path, 'fig', event))
        with open(os.path.join(self.base_path, 'ori_waveform', event, 'stalst')) as f:
            lines = f.readlines()

        stat = 'SSLB'
        self.create_directory(os.path.join(self.base_path, 'fig', event, stat))
        hvv, corrv, filter3 = [], [], []

        for filter1 in range(20, 140, 10):
            print(f"It's working on period {filter1}")
            self.process_filter(event, stat, filter1, hvv, corrv, filter3)

        self.plot_results(event, stat, hvv, corrv, filter3)

    def create_directory(self, path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    def process_filter(self, event, stat, filter1, hvv, corrv, filter3):
        bp1 = 1 / (filter1 + 15)
        bp2 = 1 / (filter1 - 15)
        st1, st2, st3 = self.read_waveforms(event, stat)
        self.apply_filters(st1, st2, st3, bp1, bp2)
        tr1, tr2, tr3 = st1[0], st2[0], st3[0]
        e, n, z = tr1.data, tr2.data, tr3.data
        ori_r, ori_t = self.calculate_ori(e, n, tr1.meta.sac.baz)
        x = tr1.times()
        new_z, envelope = self.calculate_hilbert(z)
        corne, theta, RT_ratio_list = self.calculate_correlations(e, n, z, ori_r, new_z, x, tr1.meta.sac.baz, filter1)
        self.update_results(corne, theta, RT_ratio_list, e, n, z, ori_r, ori_t, tr1.meta.sac.baz, filter1, hvv, corrv, filter3)

    def read_waveforms(self, event, stat):
        st1 = obspy.read(os.path.join(self.base_path, 'ori_waveform', event, f'{event}.{stat}.BHE.sac'))
        st2 = obspy.read(os.path.join(self.base_path, 'ori_waveform', event, f'{event}.{stat}.BHN.sac'))
        st3 = obspy.read(os.path.join(self.base_path, 'ori_waveform', event, f'{event}.{stat}.BHZ.sac'))
        return st1, st2, st3

    def apply_filters(self, st1, st2, st3, bp1, bp2):
        for st in [st1, st2, st3]:
            st.filter('bandpass', freqmin=bp1, freqmax=bp2, corners=1, zerophase=True)
            st.taper(max_percentage=0.05)

    def calculate_ori(self, e, n, baz):
        ori_r = -e * sin(radians(baz)) - n * cos(radians(baz))
        ori_t = -e * cos(radians(baz)) + n * sin(radians(baz))
        return ori_r, ori_t

    def calculate_hilbert(self, z):
        h = hilbert(z)
        envelope = np.abs(h)
        new_z = -h.imag
        return new_z, envelope

    def calculate_correlations(self, e, n, z, ori_r, new_z, x, baz, filter1):
        corne, theta, RT_ratio_list = [], [], []
        p = np.argmax(np.abs(hilbert(z)))
        R1, R2 = int(x[p] - filter1 * 3), int(x[p] + filter1 * 3)
        ba = baz - 100
        while ba < baz + 100:
            r, t = self.rotate_components(e, n, ba)
            corne.append(self.calculate_correlation(r, new_z, R1, R2))
            theta.append(ba)
            RT_ratio_list.append(self.calculate_RT_ratio(r, t, R1, R2))
            ba += 5
        return corne, theta, RT_ratio_list

    def rotate_components(self, e, n, ba):
        r = -e * sin(radians(ba)) - n * cos(radians(ba))
        t = -e * cos(radians(ba)) + n * sin(radians(ba))
        return r, t

    def calculate_correlation(self, r, new_z, R1, R2):
        fct = correlate(r[R1 * 10:R2 * 10], new_z[R1 * 10:R2 * 10], 0)
        shift, value = xcorr_max(fct)
        return value

    def calculate_RT_ratio(self, r, t, R1, R2):
        rms_r = np.sqrt(np.mean(r[R1 * 10:R2 * 10] ** 2))
        rms_t = np.sqrt(np.mean(t[R1 * 10:R2 * 10] ** 2))
        return rms_r / rms_t

    def update_results(self, corne, theta, RT_ratio_list, e, n, z, ori_r, ori_t, baz, filter1, hvv, corrv, filter3):
        max_corne = max(corne)
        maxba = np.array(theta)[corne == max_corne]
        maxRT_ba = np.array(theta)[RT_ratio_list == max(RT_ratio_list)]
        rot_r, rot_t = self.rotate_components(e, n, maxRT_ba[0])
        rms_r = np.sqrt(np.mean(rot_r ** 2))
        rms_z = np.sqrt(np.mean(z ** 2))
        hvv.append(rms_r / rms_z)
        corrv.append(maxRT_ba[0])
        filter3.append(filter1)
        self.plot_intermediate_results(e, n, z, ori_r, ori_t, rot_r, rot_t, maxRT_ba, baz, filter1, corne, theta, RT_ratio_list)

    def plot_intermediate_results(self, e, n, z, ori_r, ori_t, rot_r, rot_t, maxRT_ba, baz, filter1, corne, theta, RT_ratio_list):
        x = np.arange(len(z))
        plt.figure()
        fig, ax = plt.subplots(3, 1, figsize=(5, 4), sharey=True)
        ax[0].plot(x, z, label='z_hilbert', color='b')
        ax[0].plot(x, ori_r, label='r_ori', color='r')
        ax[0].axvline(x[int(len(z) / 2)], color='gray')
        ax[0].axvline(x[int(len(z) / 2) + filter1 * 3], color='gray')
        plt.suptitle(f'rotate angle: {maxRT_ba[0] - baz}')
        ax[0].set_title('Crz_before')
        ax[1].set_title('Crz_after')
        ax[1].plot(x, z, label='z_hilbert', color='b')
        ax[1].plot(x, rot_r, label='r_rot', color='r')
        ax[2].plot(x, ori_t, label='t_ori', color='b')
        ax[2].plot(x, rot_t, label='t_rot', color='r')
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[2].legend(loc="upper right")
        fig.tight_layout()

        fig, ax = plt.subplots(2, 1, figsize=(5, 3))
        ax[0].plot(theta - baz, corne, color='b')
        ax[1].plot(theta - baz, RT_ratio_list, color='r')
        ax[0].set_xlim((-100, 100))
        ax[0].set_xlabel('rotate angle')
        ax[0].set_ylabel('Crz_after')
        ax[0].set_title(f'rotate angle: {maxRT_ba[0] - baz}')
        ax[1].set_xlabel('rotate angle')
        ax[1].set_ylabel('rms(R)/rms(T)')
        fig.tight_layout()

    def plot_results(self, event, stat, hvv, corrv, filter3):
        orihv = self.hv10[(self.hv10[6] == stat) & (self.hv10[5] == float(event))]
        plt.figure()
        fig, ax = plt.subplots(2, 1, figsize=(5, 4))
        ax[0].scatter(orihv[4], orihv[2], label='original H/V', color='gray')
        ax[0].scatter(filter3, hvv, label='corrected H/V', color='r')
        ax[0].set_title(f'rotate angle: [mean:{np.array(corrv).mean() - tr1.meta.sac.baz:.2f} std:{np.array(corrv).std():.2f}]')
        ax[0].legend()
        ax[0].set_ylabel('H/V')
        ax[1].scatter(filter3, corrv - tr1.meta.sac.baz, label='rotate angle', color='b')
        ax[1].axhline(np.array(corrv).mean() - tr1.meta.sac.baz, label='mean rotate angle', color='r')
        ax[1].legend()
        ax[1].set_ylabel('rotate angle')
        ax[1].set_xlabel('period')
        fig.tight_layout()

if __name__ == '__main__':
    processor = HVRatioCalculator(base_path='\\ok_waveform', hv_file='hv_10_2.txt')
    for event in processor.all_event[12:13]:
        processor.process_event(event)
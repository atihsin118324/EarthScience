"""
Final Project:
 1: The Keeling Curve of CO2 data (monthly since 1958)
 Primary Mauna Loa CO2 Record
 Author: IHSIN CHANG
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as colors
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import leastsq
from scipy.signal import butter, lfilter, filtfilt, freqs


def custom_formatter(x, pos):
    return f'{x:.0f}'


def plot_psd(y, window_size_t, noverlap_percentage, dt=1, figname='psd.jpg'):
    k = 3
    N = len(y)
    freqs_resolution = 1 / N
    plt.figure(figsize=(6, 4))
    plt.title(
        f'window_size: {window_size_t}  noverlap: {noverlap_percentage}')
    window_size = round(window_size_t / dt)
    noverlap = round(noverlap_percentage * window_size)
    ax = plt.gca()
    Pxx, freqs, line = ax.psd(y, NFFT=window_size, noverlap=noverlap, Fs=1 / dt,
                              c='b', return_line=True)
    # mask = np.argmax(Pxx)
    mask = Pxx.argsort()[-1 * k:][::-1]
    max_freq = freqs[mask]
    max_periods = [1 / f for f in max_freq]
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.scatter(freqs, line[0].get_ydata(), c='k', s=3)
    ax.scatter(max_freq, line[0].get_ydata()[mask], marker='x', c='r')
    ax.set_xlabel('Period (month)')
    ax.set_ylabel('Power Spectral Density (dB/month)')
    max_freq = [f'{e:.4f}' for e in max_freq]
    max_periods = [f'{e:.4f}' for e in max_periods]

    plt.title(
        f'Power Spectrum Density \n{freqs_resolution=:.5f} ({window_size=}, overlap={noverlap_percentage})\n {max_freq=}\n{max_periods=}',
        fontweight='bold', fontsize=12)

    freqs_labels = ax.get_xticks()
    periods_labels = [f"{1 / f:.1f}" if f != 0 else 'inf' for f in freqs_labels]
    ax.set_xticks(ticks=freqs_labels, labels=periods_labels)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlim(0, 0.5)

    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    plt.savefig(figname)


def plot_spectrogram(x=None, y=None, window_size_t=None,
                     noverlap_percentage=None, dt=1, figname='spectrogram.jpg'):
    N = len(y)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.set_title(
        f'Spectrogram \n(window_size: {window_size_t}, noverlap: {noverlap_percentage})',
        fontweight='bold', fontsize=12)
    window_size = round(window_size_t / dt)
    noverlap = round(noverlap_percentage * window_size)

    spectrum, freqs, t, im = ax.specgram(y, NFFT=window_size, noverlap=noverlap,
                                         Fs=1 / dt,
                                         cmap='jet')

    freqs_labels = ax.get_yticks()
    periods_labels = [f"{1 / f:.1f}" if f != 0 else 'inf' for f in freqs_labels]
    ax.set_yticks(ticks=freqs_labels, labels=periods_labels)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax.set_xlabel('Time (month)')
    ax.set_ylabel('Period (month)')
    cbar_ax = inset_axes(ax, width="60%", height="2%", loc='lower right',
                         bbox_to_anchor=(0.5, -0.1, 0.5, 1.0),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
    colorbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    colorbar.set_label('PSD (dB)')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
    plt.savefig(figname)


def plot_power_spectrum(y):
    k = 3
    plt.figure(figsize=(6, 4))
    N = len(y)
    dt = 1
    T = N * dt
    xf = fftfreq(N, dt)
    ps = np.abs(fft(y)) ** 2
    freqs_resolution = 1 / N
    mask = np.argsort(ps[0:N // 2])[::-1][:k]
    max_freq = xf[:N // 2][mask]
    max_periods = [1 / f for f in max_freq]
    max_freq = [f'{e:.4f}' for e in max_freq]
    max_periods = [f'{e:.4f}' for e in max_periods]

    plt.plot(xf[:N // 2], ps[0:N // 2], c='b')
    plt.scatter(xf[:N // 2], ps[0:N // 2], c='k', s=3)
    # plt.scatter(max_freq, ps[mask], marker='x', c='r')
    ax = plt.gca()
    freqs_labels = ax.get_xticks()
    freqs_labels = freqs_labels[1:-1]
    periods_labels = [f"{1 / f:.1f}" if f != 0 else 'inf' for f in freqs_labels]
    ax.set_xticks(ticks=freqs_labels, labels=periods_labels)
    ax.set_xlim(0, 0.5)
    plt.grid()
    plt.xlabel('Period (month)')
    plt.ylabel('P.S. (ppm$^{2}$)')
    plt.title(
        f'Power Spectrum ({freqs_resolution=:.5f})\n {max_freq=}\n{max_periods=}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('PS.jpg')


def plot_wavelet_spectrum(wavelet_name, x=None, y=None, scales=None):
    coef, freqs = pywt.cwt(y, scales, wavelet_name)
    power = np.abs(coef) ** 2
    vmin, vmax = power.min(), power.max()
    levels = np.linspace(vmin, vmax, num=100)
    plt.figure(figsize=(8, 6))

    # norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=1)
    im = plt.contourf(x, freqs, power, cmap='jet', vmin=vmin, vmax=vmax,
                      levels=levels)
    plt.yscale('log', base=2)
    plt.title(
        f'CWT Wavelet (time-frequency) spectrum \n(wavelet type: {wavelet_name})',
        fontweight='bold',
        fontsize=18)
    plt.xlabel('Time (month)', fontsize=15)
    plt.ylabel('Period (month)', fontsize=15)
    ax = plt.gca()
    cbar_ax = inset_axes(ax, width="40%", height="2%", loc='lower right',
                         bbox_to_anchor=(0.5, -0.18, 0.5, 1.0),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
    colorbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    colorbar.set_label('Power (ppm$^{2}$)')
    tick_values = np.linspace(vmin, vmax, 3)
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels([f'{tick:.2f}' for tick in tick_values])
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_tick_params(rotation=45)

    ylabels = ax.get_yticks()
    ylabels = ylabels[1:-1]
    periods = [f'{1 / f:.2f}' if f != 0 else '' for f in ylabels]
    ax.set_yticks(ticks=ylabels, labels=periods)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    plt.savefig('wavelet_analysis.jpg', dpi=300)


def plot_basic(df):
    df.columns = ['Yr', 'Mn', 'Date Excel', 'Date', 'CO2',
                  'seasonally adjusted',
                  'fit', 'seasonally adjusted fit', 'CO2 filled',
                  'seasonally adjusted filled', 'Sta']
    x = df['Date']
    y = df['CO2 filled']
    fit = df['fit']
    seasonally_adjusted_fit = np.array(df['seasonally adjusted fit'])
    seasonally_adjusted_filled = np.array(df['seasonally adjusted filled'])
    resid = seasonally_adjusted_filled - seasonally_adjusted_fit

    plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    plt.plot(x, y, c='k', label='CO$_{2}$ filled')
    plt.plot(x, fit, c='r',
             label='fit (a stiff cubic spline function plus 4-harmonic)')
    plt.plot(x, seasonally_adjusted_filled, c='b',
             label='seasonally adjusted filled (subtracting a 4-harmonic fit with a linear gain factor)')
    plt.plot(x, seasonally_adjusted_fit, c='g', label='seasonally adjusted fit')

    plt.xlabel('Time (month)')
    plt.ylabel('CO$_{2}$ concentration (ppm)')
    plt.title('Carbon dioxide concentration at Mauna Loa Observatory*',
              fontweight='bold', fontsize=12)
    plt.legend()
    plt.grid()
    plt.ylim(300, 450)

    plt.subplot(3, 1, 2)
    plt.plot(x, y, c='k', label='CO$_{2}$ filled')
    plt.plot(x, fit, c='r',
             label='fit (a stiff cubic spline function plus 4-harmonic)')
    plt.plot(x, seasonally_adjusted_filled, c='b',
             label='seasonally adjusted filled (subtracting a 4-harmonic fit with a linear gain factor)')
    plt.plot(x, seasonally_adjusted_fit, c='g', label='seasonally adjusted fit')
    plt.xlabel('Time (month)')
    plt.ylabel('CO$_{2}$ concentration (ppm)')
    plt.title('Zoom in', fontweight='bold', fontsize=12)
    plt.grid()
    plt.legend()
    plt.xlim(2016, 2020)
    plt.ylim(400, 420)

    plt.subplot(3, 1, 3)
    plt.plot(x, resid, c='k')
    plt.ylabel('CO$_{2}$ concentration (ppm)')
    plt.title(
        'CO$_{2}$ residual \n (seasonally adjusted filled - seasonally adjusted fit)',
        fontweight='bold', fontsize=12)
    plt.grid()

    plt.tight_layout()
    plt.savefig('data_residual.jpg')
    plt.show()


def plot_compare(x, y1, y2):
    plt.plot(x, y1, c='k', label='remove the quasi-regular seasonal cycle')
    plt.plot(x, y2, c='r',
             label='remove LS fit (t, t$^{2}$, seasonal, half-seasonal)')
    plt.ylabel('CO$_{2}$ concentration (ppm)')
    corr = signal.correlate(y1, y2)
    corr_normalized = corr / (np.std(y1) * np.std(y2) * len(y1))
    corr_value = np.max(corr_normalized)
    plt.title(
        f'Compare two methods for CO$_{2}$ residual ({corr_value=:.2f})',
        fontweight='bold', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig('compare_residual.jpg')
    plt.show()


def plot_all_analysis(x, y_resid):
    scales = np.arange(1, 64)
    wavelet_name = 'cmor1.5-1.0'
    window_size_t = 150
    noverlap_percentage = 0.3
    plot_psd(y_resid,
             window_size_t=window_size_t,
             noverlap_percentage=noverlap_percentage,
             figname='psd.jpg')
    plot_spectrogram(x, y_resid, window_size_t=window_size_t,
                     noverlap_percentage=noverlap_percentage, dt=1,
                     figname='spectrogram.jpg')
    plot_power_spectrum(y_resid)
    plot_wavelet_spectrum(wavelet_name, x, y_resid, scales)
    plt.show()


def process_ori_residual(plot=True):
    input_file = "monthly_in_situ_co2_mlo.csv"
    df = pd.read_csv(input_file, header=None, skiprows=range(60))
    df.columns = ['Yr', 'Mn', 'Date Excel', 'Date', 'CO$_{2}$',
                  'seasonally adjusted',
                  'fit', 'seasonally adjusted fit', 'CO$_{2}$ filled',
                  'seasonally adjusted filled', 'Sta']

    # remove -99.99 start and end
    print('Remove -99.99 for index:')
    print(df[df['seasonally adjusted fit'] == -99.99].index)
    start_index = df[df['seasonally adjusted fit'] != -99.99].index[0]
    end_index = df[df['seasonally adjusted fit'] != -99.99].index[-1]
    df_filtered = df.loc[start_index:end_index]
    # interpolate -99.99 in the middle
    df_filtered.replace(-99.99, np.nan, inplace=True)
    df_filtered.interpolate(inplace=True)

    x = np.array(df_filtered['Date'])
    y = np.array(df_filtered['CO$_{2}$ filled'])
    y_fit = np.array(df_filtered['fit'])
    seasonally_adjusted_fit = np.array(df_filtered['seasonally adjusted fit'])
    seasonally_adjusted_filled = np.array(
        df_filtered['seasonally adjusted filled'])
    y_resid = seasonally_adjusted_filled - seasonally_adjusted_fit
    y_detrend = LSM(x, y)

    if plot:
        plot_basic(df)
        plot_all_analysis(x, y_resid)
        # plot_all_analysis(x, y_detrend)

    return x, y_resid


def process_LS_residual(plot=True):
    x1, y_resid1 = process_ori_residual(plot=False)
    input_file = "residual_selfprocess_2_lstqr.csv"
    df = pd.read_csv(input_file)
    x = np.array(df['Date'])
    y_resid2 = np.array(df['Residual'])
    if plot:
        plot_compare(x, y_resid1, y_resid2)
        plot_all_analysis(x, y_resid2)


def f_seasonal2(m, t):
    return m[0] + (m[1] * t) + (m[2] * (t ** 2)) + (
                m[3] * np.cos(2 * np.pi * t / 12)) + (
            m[4] * np.sin(2 * np.pi * t / 12)) + (
                m[5] * np.cos(2 * np.pi * t / 6)) + (
            m[6] * np.sin(2 * np.pi * t / 6)) + (
                m[7] * np.cos(2 * np.pi * t / 4)) + (
            m[8] * np.sin(2 * np.pi * t / 4)) + (
                m[9] * np.cos(2 * np.pi * t / 3)) + (
            m[10] * np.sin(2 * np.pi * t / 3))


def f_seasonal(m, t):
    return m[0] + (m[1] * t) + (m[2] * (t ** 2)) + (
                m[3] * np.cos(2 * np.pi * t / 12)) + (
            m[4] * np.sin(2 * np.pi * t / 12))


def f_trend(m, t):
    return m[0] + (m[1] * t) + (m[2] * (t ** 2))


def residuals(m, t, y):
    ret = f_trend(m, t) - y
    return ret


def LSM(x, y):
    m_init = np.random.rand(3)
    m_lsq = leastsq(residuals, m_init, args=(x, y))

    x_lsq = np.linspace(np.min(x), np.max(x), 1000)
    y_lsq = f_trend(m_lsq[0], x_lsq)
    res = residuals(m_lsq[0], x, y)
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(x, y, c='k', label='d_obs')
    # plt.plot(x_lsq, y_lsq, c='r', label='d_predict')
    # plt.xlabel('Time (days)')
    # plt.ylabel('Amplitude (mm)')
    # plt.legend()
    # s = [f'{m:.3f}' for m in m_lsq[0]]
    # plt.title(f'm={s}')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x, res, c='k')
    # plt.xlabel('Time (days)')
    # plt.ylabel('Residual Amplitude (mm)')
    # plt.title(f'Total error: {np.sum(res)}')
    #
    # plt.tight_layout()
    # plt.savefig('LS.png')
    # plt.show()

    return res


def bw_highpass(x, wn, order=4):
    b, a = butter(order, wn, btype='highpass')
    filtered_signal = filtfilt(b, a, x)
    return filtered_signal


def bw_lowpass(x, wn, order=4):
    b, a = butter(order, wn, btype='lowpass')
    filtered_signal = filtfilt(b, a, x)
    return filtered_signal


def process_enso():
    input_file = "ENSO_process.csv"
    df = pd.read_csv(input_file)
    y = np.array(df['value'])
    x = np.arange(len(y))
    plot_all_analysis(x, y)


def main():
    process_ori_residual()
    # process_LS_residual()
    # process_enso()


if __name__ == '__main__':
    main()

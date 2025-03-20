"""
Practice for Fast Fourier Transform (FFT).
1. Plot the peridogram (power or square of amplitude) as a function of frequency.
2. Apply the inverse FFT to obtain the filtered signal in time domain.
3. Identify these two dominant frequencies and apply the inverse FFT.
4. Convolution of two boxcar functions
5. Bandpass filter
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
import obspy


class FFTProcessor:
    def __init__(self, file_path, fs=1):
        self.file_path = file_path
        self.fs = fs
        self.intensity = self.load_data()
        self.f = None
        self.Pxx_den = None
        self.sig_fft = None
        self.power = None
        self.freq = None

    def load_data(self):
        return np.loadtxt(self.file_path)

    def calculate_periodogram(self):
        N = self.intensity.size
        self.f, self.Pxx_den = scipy.signal.periodogram(self.intensity, self.fs)

        self.sig_fft = scipy.fftpack.fft(self.intensity)
        self.power = 2 / N * np.abs(self.sig_fft) ** 2
        self.freq = scipy.fftpack.fftfreq(N)

        plt.figure()
        plt.semilogy(self.f, self.Pxx_den, label='signal.periodogram')
        plt.semilogy(self.freq[:N // 2], self.power[:N // 2], label='fftpack.fft')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V**2/Hz)')
        plt.ylim([0.01, 100000])
        plt.legend()
        plt.show()

    def filter_signal(self, threshold=1000):
        N = len(self.sig_fft)
        main_freq_fft = self.sig_fft.copy()
        main_freq_fft[self.power < threshold] = 0
        filtered_sig = scipy.fftpack.ifft(main_freq_fft)

        plt.figure()
        plt.plot(np.arange(N), self.intensity, c='lightgray', label='original')
        plt.plot(np.arange(N), filtered_sig, c='b', label='filtered')
        plt.ylabel('Intensity')
        plt.xlabel('Time (day)')
        plt.legend()
        plt.show()

        return filtered_sig

    def filter_signal_two_dominant(self):
        N = len(self.sig_fft)
        freq_zero = np.where(self.freq == 0)
        sorted_power = np.sort(self.Pxx_den)[::-1]
        sorted_freq = self.f[self.Pxx_den.argsort()[::-1]]
        mask = [i for i, p in enumerate(self.power) if
                p not in sorted_power[:2] and p != self.power[freq_zero]]
        main_freq_fft = self.sig_fft.copy()
        main_freq_fft[mask] = 0
        filtered_sig = scipy.fftpack.ifft(main_freq_fft)

        print('PSD', sorted_power[:5])
        print('f', sorted_freq[:5])

        plt.figure()
        plt.plot(np.arange(N), self.intensity, c='lightgray', label='original')
        plt.plot(np.arange(N), filtered_sig, c='b', label='filtered')
        plt.ylabel('Intensity')
        plt.xlabel('Time (day)')
        plt.legend()
        plt.show()

        return filtered_sig

    def run(self):
        self.calculate_periodogram()
        self.filter_signal()
        self.filter_signal_two_dominant()


class BoxcarConvolution:
    def __init__(self, TD):
        self.TD = TD
        self.TD_sig = np.pad(scipy.signal.get_window('boxcar', TD), (TD, TD))

    def generate_boxcar_signal(self, theta):
        TR = self.TD * (1/0.8 - np.cos(np.deg2rad(theta)))
        TR = round(TR)
        TR_sig = np.pad(scipy.signal.get_window('boxcar', TR), (0, 3*self.TD-TR))
        return TR_sig

    def convolve_signals(self, TR_sig):
        return scipy.signal.convolve(self.TD_sig, TR_sig, mode='full') / sum(TR_sig)

    def plot_signals(self, TR_sig, fbox, theta):
        plt.figure()
        plt.title(f'theta = {theta}')
        plt.plot(self.TD_sig, label='signal (TD)', lw=2, c='gray')
        plt.plot(TR_sig, label='box (TR)', lw=2, c='k')
        plt.plot(fbox, label='convolve', lw=2, c='r')
        plt.ylim([0, 1.5])
        plt.legend()

    def run(self):
        for theta in range(0, 360+45, 45):
            TR_sig = self.generate_boxcar_signal(theta)
            fbox = self.convolve_signals(TR_sig)
            self.plot_signals(TR_sig, fbox, theta)
        plt.show()


class BandpassFilter:
    def __init__(self, file_path):
        self.tr = obspy.read(file_path)[0]
        self.signal = self.tr.data.copy()
        self.delta = self.tr.stats.delta
        self.N = self.tr.stats.npts
        self.t = self.delta * np.arange(self.N)
        self.sig_fft = scipy.fftpack.fft(self.signal)
        self.power = 2 / self.N * np.abs(self.sig_fft) ** 2
        self.freq = scipy.fftpack.fftfreq(self.N)
        self.f1 = None
        self.f2 = None
        self.sig_fft1 = None
        self.power1 = None
        self.sig_fft2 = None
        self.power2 = None
        self.F1 = None
        self.F2 = None

    def bandpass_filter(self):
        self.f1 = self.tr.copy().filter('bandpass', freqmin=0.005, freqmax=0.008, corners=1, zerophase=True)
        self.f2 = self.tr.copy().filter('bandpass', freqmin=0.001, freqmax=0.004, corners=1, zerophase=True)
        self.sig_fft1 = scipy.fftpack.fft(self.f1)
        self.power1 = 2 / self.N * np.abs(self.sig_fft1) ** 2
        self.sig_fft2 = scipy.fftpack.fft(self.f2)
        self.power2 = 2 / self.N * np.abs(self.sig_fft2) ** 2
        self.F1 = abs(self.power1 * self.power2)
        self.F2 = abs(self.power1) * abs(self.power2)

    def plot_filtered_signals(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.f1, label='f1 (0.005-0.008 Hz)')
        plt.plot(self.t, self.f2, label='f2 (0.001-0.004 Hz)')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.title('filtered')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.freq[:self.N // 2], self.F1[:self.N // 2], label='F1')
        plt.plot(self.freq[:self.N // 2], self.F2[:self.N // 2], label='F2')
        plt.axvline(x=2*10**(-4), ymin=0, ymax=1, linestyle='--', c='gray', label=r'$2*10^{-4}$')
        plt.axvline(x=5*10**(-4), ymin=0, ymax=1, linestyle='--', c='k', label=r'$5*10^{-4}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.filter_signals()
        self.plot_filtered_signals()


if __name__ == '__main__':
    processor = FFTProcessor(file_path='varstar.dat')
    processor.run()
    boxcar_convolution = BoxcarConvolution(TD=100)
    boxcar_convolution.run()
    bp_filter = BandpassFilter(file_path='IU.ANMO..BHE.sac')
    bp_filter.run()

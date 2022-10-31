import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

c = 3.0e8  # m/s
wave_length_0 = 560.0e-9
h = 6.626e-34  # J*s
k = 1.38e-23  # J/K


def spectrum_power_wave_length(wave_length, T):
    ret = 2.0 * np.pi * h * c * \
          np.power(wave_length, -3) * 1.0 / (np.exp(h * c / (k * T * wave_length)) - 1)
    return ret


def d_spectrum_power_wave_length(wave_length, T):
    ret = 2.0 * np.pi * h * c * \
          np.power(wave_length, -4) * 1.0 / (np.exp(h * c / (k * T * wave_length)) - 1) * \
          (-3 + (h * c / (k * T)) * np.exp(h * c / (k * T * wave_length)) / \
           (wave_length * (np.exp(h * c / (k * T * wave_length)) - 1)))
    return ret


def spectrum_power_nu(nu, T):
    ret = (2.0 * np.pi * np.power(nu, 3) / np.power(c, 2)) * 1.0 / (np.exp(h * nu / (k * T)) - 1)
    return ret


def wafe_freq(wave_length):
    ret = c / wave_length
    return ret


wave_length_nm = np.arange(100.0, 1000.0, 0.5)
wave_length_nm_filters = np.array([488.0, 533.0, 632.8])
#wave_length_nm_filters = np.array([380.0, 533.0, 740.0])
wave_length = wave_length_nm * 1.0e-9
wave_length_filters = wave_length_nm_filters * 1.0e-9
# plasma_T_eV = np.arange(1.0, 3.5, 0.5)
# plasma_T = plasma_T_eV * 11604.5
plasma_T = np.arange(2000, 4500, 1000)
'''for i, T1 in enumerate(plasma_T):
    for j, T2 in enumerate(plasma_T):
        if j > i:
            x = wave_length_nm
            y = spectrum_power(wafe_freq(wave_length), T2) - spectrum_power(wafe_freq(wave_length), T1)
            y_mean = 0.5*(spectrum_power(wafe_freq(wave_length), T2) + spectrum_power(wafe_freq(wave_length), T1))
            y = y/y_mean
            plt.plot(x, y, label=f'R({T2} K)-R({T1} K)')
            plt.fill_between(x,y,
                             where=(x > 380) & (x <= 740),
                             color='g')'''


def R0(T):
    return spectrum_power_nu(wafe_freq(wave_length_0), T)


for i, T in enumerate(plasma_T):
    x = wave_length_nm
    x_filter = wave_length_nm_filters
    y = spectrum_power_nu(wafe_freq(wave_length), T)
    y_filter = spectrum_power_nu(wafe_freq(wave_length_filters), T)

    poly = np.polyfit(x_filter, y_filter, 1)
    poly_f = np.poly1d(poly)
    poly_f_plot = poly_f(x)
    poly_m = np.polyfit(x_filter*1.0e-9, y_filter, 1)
    f_lambda = lambda t: R0(t) - poly_m[-1]
    Troot = fsolve(f_lambda,[1000,4000])
    print(f'{Troot}; T={T}')

    # y=y/np.max(y)
    plt.plot(x, y, label=f'T = {T} K')
    plt.plot(x_filter, y_filter, 'o')
    plt.plot(x, poly_f(x))
    plt.fill_between(x, y,
                     where=(x > 380) & (x <= 740),
                     color='y')

plt.title('Absolutly black')
plt.xlabel('$\lambda, nm$')
plt.ylabel('Spectr')
plt.legend()
plt.grid()
plt.show()

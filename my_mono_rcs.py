from matplotlib import pyplot as plt
import numpy as np
from DielectricMaterial import DielectricMaterial
from getRCS import Bistatic_RCS, RCS_vs_freq, plotBiRCS, plotOneMonoRCS
from getDielectricSphereFieldUnderPlaneWave import *
from DielectricMaterial import *
from src import *
import os

if __name__ == '__main__':
    air = DielectricMaterial(1, 0)
    sphere_material = DielectricMaterial(3, 0, 2, 0)
    radius = 0.1
    freq_max = 5e9
    freq_min = 1e7
    freq_step = 0.03549593050601196e9
    freq = np.arange(freq_min, freq_max, freq_step)
    wavelength = 3e8 / freq
    ratio = radius / wavelength

    freq, mono_rcs = RCS_vs_freq(radius=0.1, ratio=ratio, background_material=air,
                                 sphere_material=sphere_material, sensor_location=[0, 0, -2000], show_plot=False)

    # TEST

    phi = 0
    distance = 2000

    sensor_location = sphToCart(distance, np.pi, phi)
    [E_r, E_theta, E_phi, H_r, H_theta, H_phi] = \
        getDielectricSphereFieldUnderPlaneWave(
            radius, sphere_material, air, sensor_location, freq)
    E = (np.stack((E_r, E_theta, E_phi), axis=0))

    bit_to_mono_rcs = 4*np.pi * (norm(sensor_location)**2) * \
        np.sum((E * np.conj(E)), axis=0)
    print(bit_to_mono_rcs.shape)

    # plot
    f, ax = plt.subplots(2, 1, figsize=(6, 10))
    ax[0].plot(freq/1e9, 10*np.log10(np.abs(mono_rcs)), label='mono')
    ax[0].plot(freq/1e9, 10*np.log10(np.abs(bit_to_mono_rcs)), label='bit')
    ax[0].set_xlabel('Frequency (GHz)')
    ax[0].set_ylabel('RCS')
    ax[0].legend()
    ax[0].grid()

    err = np.abs(mono_rcs) - np.abs(bit_to_mono_rcs)
    ax[1].plot(freq/1e9, err)
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].set_ylabel('Error')
    ax[1].grid()

    plt.show()

    np.save('mie_mono_rcs.npy', np.stack((freq, mono_rcs), axis=0))
    np.save('mie_bit_to_mono_rcs.npy', np.stack(
        (freq, bit_to_mono_rcs), axis=0))

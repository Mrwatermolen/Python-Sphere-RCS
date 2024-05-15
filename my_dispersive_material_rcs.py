from matplotlib import pyplot as plt
import numpy as np
from DielectricMaterial import DielectricMaterial
from getRCS import Bistatic_RCS
import os


from my_dispersive_model import MLorentzModel, DrudeModel, DebyeModel, dispersive_material_mono_rcs, air, EPSILON_0

if __name__ == '__main__':
    air = DielectricMaterial(1, 0)

    omega_p = 2*np.pi*2e9
    nv = np.pi*2e9
    epsilon_r_inf = 2
    epsilon_r_static = 5
    lorentz = MLorentzModel(
        name='lorentz',
        epsilon_r_inf=2,
        a_0=[(epsilon_r_static - epsilon_r_inf) * omega_p**2],
        a_1=[0],
        b_0=[omega_p**2],
        b_1=[2*nv],
        b_2=[1]
    )

    epsilon_r_inf = 2
    epsilon_r_static = 7
    tau = 2e-9 / (2 * np.pi)
    deby = DebyeModel(
        name='deby',
        epsilon_r_inf=epsilon_r_inf,
        epsilon_s=[epsilon_r_static],
        tau=[tau]
    )

    drdue = DrudeModel(
        name='drdue',
        epsilon_r_inf=4,
        omega_p=[np.pi * 1e9],
        gamma=[np.pi * 1.2e9]
    )

    radius = 0.1
    frequency = np.linspace(5e7, 5e9, 100)

    mono_rcs = dispersive_material_mono_rcs(
        drdue, radius=radius, freq=frequency, distance=2000)

    freq = 1e9
    theta, bistatic_rcs = Bistatic_RCS(
        radius=radius, frequency=freq, background_material=air, sphere_material=drdue.normalMaterial(freq), distance=2000, phi=0, show_plot=False)

    np.save('mie_mono_rcs.npy', np.stack((frequency, mono_rcs), axis=0))
    np.save('mie_bistatic_rcs.npy', np.stack((theta, bistatic_rcs), axis=0))
    
    f,ax = plt.subplots()
    ax.plot(frequency, 10*np.log10(np.abs(mono_rcs)), label='Mono')
    plt.show()

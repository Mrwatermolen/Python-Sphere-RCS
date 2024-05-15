from matplotlib import pyplot as plt
from getRCS import Bistatic_RCS
from my_dispersive_model import MLorentzModel, DrudeModel, DebyeModel, dispersive_material_mono_rcs, air, EPSILON_0
import numpy as np

if __name__ == '__main__':
    ag_drude = DrudeModel(
        name='ag_drude',
        epsilon_r_inf=3.7,
        omega_p=[1.3825e+16],
        gamma=[2.7347e+13]
    )
    
    radius = 40e-9
    freq_unit = 1e12
    frequency = np.linspace(300, 1200, 120) * freq_unit
    
    mono_rcs = dispersive_material_mono_rcs(
        ag_drude, radius=radius, freq=frequency, distance=2e-6)
    
    freq = 800*freq_unit
    theta, bistatic_rcs = Bistatic_RCS(
        radius=radius, frequency=freq, background_material=air, sphere_material=ag_drude.normalMaterial(freq), distance=2e-6, phi=0, show_plot=False)
    
    np.save('ag_drude_mie_mono_rcs.npy', np.stack((frequency, mono_rcs), axis=0))
    np.save('ag_drude_mie_bistatic_rcs.npy', np.stack((theta, bistatic_rcs), axis=0))
    

    # # plot mono
    # f,ax = plt.subplots()
    # ax.plot(frequency/freq_unit, 10*np.log10(np.abs(mono_rcs)), label='Mono')
    # ax.set_xlabel('Frequency (THz)')
    # ax.set_ylabel('Mono RCS (dB)')
    # ax.set_ylim([-170, -130])
    # ax.legend()
    # ax.grid()
    
    # # plot bistatic
    # f,ax = plt.subplots()
    # ax.plot(theta, 10*np.log10(np.abs(bistatic_rcs)), label='Bistatic')
    # ax.set_xlabel('Theta (degree)')
    # ax.set_ylabel('Bistatic RCS (dB)')
    # ax.legend()
    # ax.grid()
    
    # plt.show()

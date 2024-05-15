from DielectricMaterial import DielectricMaterial
from getDielectricSphereFieldUnderPlaneWave import getDielectricSphereFieldUnderPlaneWave
import numpy as np
from src import norm, sphToCart

air = DielectricMaterial(1, 0)

EPSILON_0 = 8.854187817e-12

class LinearDispersiveModel:
    def __init__(self, name: str, epsilon_r_inf: float, num_poles: int) -> None:
        self.name = name
        self.epsilon_r_inf = epsilon_r_inf
        self.num_poles = num_poles

    def subceptibility(self, freq, p):
        pass

    def epsilon_r(self, freq) -> np.complex128:
        return self.epsilon_r_inf + \
            sum([self.subceptibility(freq, p) for p in range(self.num_poles)])

    def normalMaterial(self, freq):
        omega = 2*np.pi*freq
        eps_r = self.epsilon_r(freq)
        sigma_e = -omega * eps_r.imag * EPSILON_0
        return DielectricMaterial(epsilon_r=eps_r.real, mu_r=1, sigma_e=sigma_e, sigma_m=0)

    def to_string(self, freq):
        brief = f'{self.name} in freq {freq} Hz'
        m = self.normalMaterial(freq)
        brief += f' with epsilon_r = {m.epsilon_r}, sigma_e = {m.sigma_e}, sigma_m = {m.sigma_m}'
        return brief


class MLorentzModel(LinearDispersiveModel):
    def __init__(self, name: str, epsilon_r_inf: float, a_0, a_1, b_0, b_1, b_2):
        super().__init__(name, epsilon_r_inf, len(a_0))
        self.a_0 = a_0
        self.a_1 = a_1
        self.b_0 = b_0
        self.b_1 = b_1
        self.b_2 = b_2

    def subceptibility(self, freq, p):
        omega = 2*np.pi*freq
        return (self.a_1[p] * omega + self.a_0[p]) / \
            (self.b_2[p] * (1j*omega)**2 +
             self.b_1[p] * 1j*omega + self.b_0[p])


class DrudeModel(LinearDispersiveModel):
    def __init__(self, name: str, epsilon_r_inf: float, omega_p: np.ndarray, gamma: np.ndarray):
        super().__init__(name, epsilon_r_inf, len(omega_p))
        self.omega_p = omega_p
        self.gamma = gamma

    def subceptibility(self, freq, p):
        omega = 2*np.pi*freq
        return -self.omega_p[p]**2 / (omega*(omega - 1j*self.gamma[p]))


class DebyeModel(LinearDispersiveModel):
    def __init__(self, name: str, epsilon_r_inf: float, epsilon_s: np.ndarray, tau: np.ndarray):
        super().__init__(name, epsilon_r_inf, len(epsilon_s))
        self.epsilon_s = epsilon_s
        self.tau = tau

    def subceptibility(self, freq, p):
        omega = 2*np.pi*freq
        return (self.epsilon_s[p] - self.epsilon_r_inf) / (1 + 1j*omega*self.tau[p])


def dispersive_material_mono_rcs(dispersive_material: LinearDispersiveModel, radius: float, freq: np.ndarray, distance: float):
    mono_rcs = np.zeros_like(freq, dtype=np.complex128)
    for i in range(len(freq)):
        f = freq[i]

        sphere_material = dispersive_material.normalMaterial(f)
        sensor_location = sphToCart(distance, np.pi, 0)
        [E_r, E_theta, E_phi, H_r, H_theta, H_phi] = \
            getDielectricSphereFieldUnderPlaneWave(
            radius, sphere_material, air, sensor_location, [f])
        E = (np.stack((E_r, E_theta, E_phi), axis=0))

        rcs = 4*np.pi * (norm(sensor_location)**2) * \
            np.sum((E * np.conj(E)), axis=0)

        mono_rcs[i] = rcs

    return mono_rcs

import numpy as np
import torch
import models
import utils

rng = np.random.default_rng()
step_rng = rng

def SGM_generation():
    #setattr(__main__, "ScoreNet", models.ScoreNet)
    score_model2 = torch.load('./saved_models/variable_dwell/global_step_001000.pth')
    y = torch.full((10,), 5)
    trained_score = lambda x, t: score_model2(x.float(), t.float(), y.int())
    samples = utils.reverse_sde(step_rng, 2, 10, utils.drift, utils.diffusivity, trained_score)
    initial_conditions = samples
    return initial_conditions

def simulate_trajectory_coupled(n, kappa, init_num, init_conds):
        epsilon: float = 1/200          # Inverse temperature.
        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1e-2                # Time step length.
        kp: float = kappa
        cond_var: float = 5.0
        h, k = 2, 0                           #The temperature that we are running at scales the barrier height by 2
        init = init_conds[init_num,:]
        def vector_field(x: np.ndarray) -> np.ndarray:  
            return np.array([1e-4 - (kp/2)*(x[0]-cond_var),
                             -((1 + x[1])**2 * (2*(1 + h - k)*x[1] - 2*h + 3*(0.75*k - 2)*x[1]**2 + 4*x[1]**3) + 
                               2*(1 + x[1])*(h - 2*h*x[1] + (1 + h - k)*x[1]**2 + (0.75*k - 2)*x[1]**3 + x[1]**4))])
        scale_param = np.array([1e-4, 1e-1]) #the temperature here corresponds to epsilon, but the sqrt etc is already taken
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,:] = init
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i,:] = x_prev + vector_field(x_prev) * dt + noise
        return xs[::1,:] 
    
def potential(x):
    h, k = 5,0  #manually set the well depth and height
    return ((x + 1)**2 *(x**4 + (0.75 * k - 2) * x**3 + (h - k + 1) * x**2 - 2 * h * x + h))
    
def true_pdf(x):
    exp_minus_potential_x = np.exp(-potential(x))
    return exp_minus_potential_x / np.trapz(exp_minus_potential_x, x=x)

    
def harmonic(kappa, xstar):
    r"""
    Returns umbrella potential
    $U(x) = \frac{\kappa}{2} (x - x^*)^2$
    Args:
        kappa (float)
        xstar (float)
    """
    def potential(x):
        return kappa / 2 * (x - xstar) ** 2
    return potential

def simulate_trajectory_US(n, kappa):
        epsilon: float = 1/200          # Inverse temperature.
        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1e-2                # Time step length.
        kp: float = kappa
        cond_var: float = 5.0
        h, k = 5, 0 
        init = [0,-1]
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-4 - (kp/2)*(x[0]-cond_var),
                             -((1 + x[1])**2 * (2*(1 + h - k)*x[1] - 2*h + 3*(0.75*k - 2)*x[1]**2 + 4*x[1]**3) + 
                               2*(1 + x[1])*(h - 2*h*x[1] + (1 + h - k)*x[1]**2 + (0.75*k - 2)*x[1]**3 + x[1]**4))])
        scale_param = np.array([1e-4, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,:] = init
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i,:] = x_prev + vector_field(x_prev) * dt + noise
        #X1 = xs[:, 0]
        return xs[::1,:]
    
kappa = 50
x_bin = np.linspace(-1.5, 1.5, 201)

def calc_norm(x2_data):
    est_pdf, est_bins = np.histogram(np.hstack(x2_data), bins=x_bin, density=True)
    dbin = est_bins[1] - est_bins[0]
    x = (est_bins + dbin / 2)[:-1]
    l1_norm = np.trapz(np.abs(est_pdf - true_pdf(x)), x=x)
    return l1_norm

def simulate_coupled_trajs(n):
    norms = []
    x_it_all = []
    US_2_all = []
    for j in range(1000):
        x_it = []
        US_2 = []
        init_conds = SGM_generation() #GAN generation of init_conds
        for i in range(10):
            US_1 = simulate_trajectory_coupled(n//10, kappa, i, init_conds)
            US_2.append(US_1[:,:])
            x_it.append(US_1[:,1])
        norm = calc_norm(x_it)
        norms.append(norm)
        x_it_all.append(x_it)
        US_2_all.append(US_2)
    return US_2_all, x_it_all, norms

def simulate_US_trajs(n):
    norms = []
    x_it_all = []
    US_2_all = []
    for j in range(1000):
        x_it = []
        US_2 = []
        for i in range(1):
            US_1 = simulate_trajectory_US(n, kappa)
            US_2.append(US_1[:,:])
            x_it.append(US_1[:,1])
        norm = calc_norm(x_it)
        norms.append(norm)
        x_it_all.append(x_it)
        US_2_all.append(US_2)
    return US_2_all, x_it_all, norms


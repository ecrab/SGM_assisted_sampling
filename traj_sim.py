import numpy as np

def evolving_dwell(n):

        num_steps: int = n               # Number of steps to integrate.
        dim: int = 2                     # Spatial dimensions.
        dt: float = 1e-2                 # Time step length.
        
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-4,
                             -(-1 + 0.2*x[0] + 4*x[1]*(-1 + x[1]**2))])
        scale_param = np.array([1e-4, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,1] = 1
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i, :] = x_prev + vector_field(x_prev) * dt + noise

        return xs 
    
    
def evolving_dwell_US(n, kappa, cv):
        # Evolving double well example with Umbrella sampling harmonic biasing term
        
        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1e-2                # Time step length.
        k: float = kappa                # Biasing constant, kappa 
        cond_var: float = cv            # Collective variable to condition or bias toward
        init = [cv,0]
        
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-4 - (k/2)*(x[0]-cond_var),
                             -(-1 + 0.2*x[0] + 4*x[1]*(-1 + x[1]**2))])
        scale_param = np.array([1e-4, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,:] = init
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i,:] = x_prev + vector_field(x_prev) * dt + noise

        return xs 
    
def variable_height_dwell(n):

        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1e-2                # Time step length.
        h, k = 2, 0                     # Variables specifiying the height of barrier and the depth of second well 
        
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-4,
                             -((1 + x[1])**2 * (2*(1 + h - k)*x[1] - 2*h + 3*(0.75*k - 2)*x[1]**2 + 4*x[1]**3) + 
                               2*(1 + x[1])*(h - 2*h*x[1] + (1 + h - k)*x[1]**2 + (0.75*k - 2)*x[1]**3 + x[1]**4))])
        scale_param = np.array([1e-4, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,0] = 0
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i, :] = x_prev + vector_field(x_prev) * dt + noise
        
        return xs 
    
def variable_height_dwell_coupled(n, kappa, init_conds, init_num, cv):
        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1e-2                # Time step length.
        k: float = kappa                # Biasing constant, kappa
        h, k = 4, 0                     # Variables specifiying the height of barrier and the depth of second well
        cond_var: float = cv            # Collective variable to condition or bias toward
        init = init_conds[init_num,:]   # Initializing with output from SGM, init_conds, and specific initial condition of
                                        # the array
        
        def vector_field(x: np.ndarray) -> np.ndarray:
            return np.array([1e-4 - (k/2)*(x[0]-cond_var),
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

        return xs[::1,:] 
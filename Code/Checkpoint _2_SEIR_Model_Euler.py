import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#PART 1 USING EULER'S


#%%
# Load the data
data = pd.read_csv("C://Users//tta20//OneDrive - University of Virginia//BME 2315 (Comp)//Module 2//Module-2-Epidemics-SIR-Modeling-Abhiram-Tahseen//Data//mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)

def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, t, N):
    """
    Solve the SEIR model using Euler's method.
    Parameters:
    - beta: Transmission rate
    - sigma: Incubation rate (1/latent period)
    - gamma: Recovery rate(1/infectious period)
    - S0: Initial susceptible population
    - E0: Intial exposed population
    - I0: Initial infected population
    - R0: Initial recovered population
    - t: Array of time points (days)
    - N: Total population
    Returns:
    - S: Array of susceptible population over time
    - E: Array of exposed population over time
    - I: Array of infected population over time
    - R: Array of recovered population over time
    """

    # Initialize arrays
    S = np.empty(len(t))
    E = np.empty(len(t))
    I = np.empty(len(t))
    R = np.empty(len(t))

    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Euler loop
    for n in range(1, len(t)):
        dt = t[n] - t[n-1]

        # Current values
        S_t = S[n-1]
        E_t = E[n-1]
        I_t = I[n-1]
        R_t = R[n-1]

        # SEIR derivatives
        dS = -beta * S_t * I_t / N
        dE = beta * S_t * I_t / N - sigma * E_t
        dI = sigma * E_t - gamma * I_t
        dR = gamma * I_t

        # Euler update
        S[n] = S_t + dS * dt
        E[n] = E_t + dE * dt
        I[n] = I_t + dI * dt
        R[n] = R_t + dR * dt

    return S, E, I, R

# PART 2 USING SSE OPTIMIZATION

# %%

def optimize_seir(timepoints, N, S0, E0, I0, R0, data):
    # Parameter ranges (these aren't the actual ranges, need to determine using our R0)
    beta_range = np.linspace(0.01, 0.02, 20) # using R0 = 0.12 from R0 = beta/gamma
    sigma_range = np.linspace(1/18, 1/12, 20)   # ≈ 0.0556 to 0.0833, using latent period 12-18 days
    gamma_range = np.linspace(1/11, 1/7, 20)   # ≈ 0.091 to 0.143, using infectious period 7-11 days

    best_sse = np.inf
    best_beta = None
    best_sigma = None
    best_gamma = None

    # Loop over all combinations
    for beta in beta_range:
        for sigma in sigma_range:
            for gamma in gamma_range:

                # Run SEIR model
                S, E, I, R = euler_seir(beta, sigma, gamma,
                                        S0, E0, I0, R0,
                                        timepoints, N)

                # Compute SSE between model I(t) and real data from release 2
                sse = np.sum((I - data)**2)

                # Track best parameters
                if sse < best_sse:
                    best_sse = sse
                    best_beta = beta
                    best_sigma = sigma
                    best_gamma = gamma

    return best_beta, best_sigma, best_gamma, best_sse


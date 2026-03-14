import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("C://Users//tta20//OneDrive - University of Virginia//BME 2315 (Comp)//Module 2//Module-2-Epidemics-SIR-Modeling-Abhiram-Tahseen//Data//mystery_virus_daily_active_counts_RELEASE#3.csv", parse_dates=['date'], header=0, index_col=None)
t = df["day"].values.astype(float)
data = df["active reported daily cases"].values


# ── PART 1: EULER SEIR ────────────────────────────────────────────────────────

def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    timepoints = np.asarray(timepoints, dtype=float)   # FIX 3: ensure numpy array

    S = np.empty(len(timepoints))
    E = np.empty(len(timepoints))
    I = np.empty(len(timepoints))
    R = np.empty(len(timepoints))

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for n in range(1, len(timepoints)):
        dt = timepoints[n] - timepoints[n-1]
        S_t, E_t, I_t, R_t = S[n-1], E[n-1], I[n-1], R[n-1]

        dS = -beta * S_t * I_t / N
        dE =  beta * S_t * I_t / N - sigma * E_t
        dI =  sigma * E_t - gamma * I_t
        dR =  gamma * I_t

        S[n] = S_t + dS * dt
        E[n] = E_t + dE * dt
        I[n] = I_t + dI * dt
        R[n] = R_t + dR * dt

    return S, E, I, R


# ── PART 2: SSE OPTIMISATION ──────────────────────────────────────────────────

def optimize_seir(t, N, S0, E0, I0, R0_init, data):
    beta_range  = np.linspace(0.45, 0.85, 20)  # using R0 = 5.74
    sigma_range = np.linspace(1/18, 1/12, 20)   # latent period 12-18 days
    gamma_range = np.linspace(1/11, 1/7,  20)   # infectious period 7-11 days

    best_sse   = np.inf
    best_beta  = None
    best_sigma = None
    best_gamma = None

    for beta in beta_range:
        for sigma in sigma_range:
            for gamma in gamma_range:
                S, E, I, R = euler_seir(beta, sigma, gamma,
                                        S0, E0, I0, R0_init,
                                        t, N)
                sse = np.sum((I - data) ** 2)
                if sse < best_sse:
                    best_sse   = sse
                    best_beta  = beta
                    best_sigma = sigma
                    best_gamma = gamma

    return best_beta, best_sigma, best_gamma, best_sse


# ── INITIAL CONDITIONS ────────────────────────────────────────────────────────

N  = 17900
S0 = N - data[0]
E0 = 0
I0 = data[0]
R0 = 0          # FIX 1: initial *recovered* count is 0, not the reproduction number


# ── RUN ───────────────────────────────────────────────────────────────────────

best_beta, best_sigma, best_gamma, best_sse = optimize_seir(
    t, N, S0, E0, I0, R0, data
)

print(f"Best beta:  {best_beta:.4f}")
print(f"Best sigma: {best_sigma:.4f}")
print(f"Best gamma: {best_gamma:.4f}")
print(f"Best SSE:   {best_sse:,.2f}")

S_fit, E_fit, I_fit, R_fit = euler_seir(
    best_beta, best_sigma, best_gamma,
    S0, E0, I0, R0, t, N
)


# ── PLOT ──────────────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.plot(t, data,  label="Data (Release 2)", linestyle='-')
plt.plot(t, I_fit, label="SEIR Model",        color='red')
plt.xlabel('Days')
plt.ylabel('Active Cases')
plt.title("Comparison of Data Release 2 with SEIR Model Predictions")
plt.legend()
plt.show()


# ── PEAK ──────────────────────────────────────────────────────────────────────

peak_day   = np.argmax(I_fit)
peak_cases = I_fit[peak_day]
print(f"Peak day: {t[peak_day]:.0f}, Peak cases: {peak_cases:.0f}")
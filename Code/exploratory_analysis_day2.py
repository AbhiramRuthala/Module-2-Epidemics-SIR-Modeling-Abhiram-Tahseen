#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv("C://Users//tta20//OneDrive - University of Virginia//BME 2315 (Comp)//Module 2//Module-2-Epidemics-SIR-Modeling-Abhiram-Tahseen//Data//mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)


# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.
popt, pcov = curve_fit(exponential_growth, data['day'], data['active reported daily cases'])

# Approximate R0 using this fit
r = popt[0]
print(f"Estimated R0: {r: .2f}")

# Add the fit as a line on top of your scatterplot.
plt.plot(data['day'], exponential_growth(data['day'], r), color = 'blue', label = 'Exponential Fit')
plt.plot(data['day'], data['active reported daily cases'], marker = 'o', color = 'red', label = 'Active Infections')

# Create the graph
plt.title('Day vs Active Infections')
plt.xlabel('Day')
plt.ylabel('Active Reported Daily Cases')
plt.legend()
plt.grid(True)

plt.show()
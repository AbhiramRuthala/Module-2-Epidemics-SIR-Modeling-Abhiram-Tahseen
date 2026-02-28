#%%

# Import all the necessary modules to do this analysis.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression

#%%
# Load the data
data = pd.read_csv('/Users/abhiramruthala/Module 2 - BME 2315/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

#plot scatterplot/lineplot of days vs. active infections - This is shown above, but is still described here because we add an exponential fit to this graph.
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['active reported daily cases'], marker='o', linestyle='-') # Line style makes the scatterplot a line plot.
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.title('Active Cases Over Time')
plt.grid()
plt.tight_layout()
plt.show()

# Put fit to the exponential growth in the above graph by sorting the data into specific x and y numpy arrays.
x = np.array(data['date'].map(pd.Timestamp.timestamp)).reshape(-1, 1)
y = np.array(data['active reported daily cases'])

# Conduct linear regression on the data by using np.polyfit
logistics = np.polyfit(x.flatten(), y, 1)
m = logistics[0] # Slope of the graph
b = logistics[1]

# Create the regression equation.
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = m * x_line + b

# Plot the same exponential graph with the fit and see how the fit aids in understanding the graph better.
plt.plot(x, y, label='Exponential Graph')
plt.plot(x_line, y_line, color='red', label='Best Fit Line')
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.title('Active Cases Over Time')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

# Use data from class resources to estimate infectious rates. Infectious rate is described as 1 / infectious period.

# In the data, it was described for the infectious period to be 2 days before symptoms appear. Our group added that as a trial
# For the R0 estimation, we added the 2 days to the symptomatic period of 5-9 days, getting 7 and 11 respectively.
infectious_Rate1 = 1/2
infectious_Rate2 = 1/7 # Trial 1 used to estimate R0
infectious_Rate3 = 1/11 # Trial 2 used to estimate R0
infectious_Rate4 = 1/9

# Since the slope of the line of best fit models how many people are being transmitted the infection per a period of time, we assumed the slope to be the beta rate.
beta_rate = m

# We applied the formula of R0 = Beta rate / infectious rate and calculated respective results.
R0_1 = beta_rate / infectious_Rate1 # Trial
R0_2 = beta_rate / infectious_Rate2 # Real 
R0_3 = beta_rate / infectious_Rate3 # Real

# R0_4 = beta_rate / infectious_Rate4
# R0_5 = beta_rate / 5

#Only do an average of the last two for an accurate R0
R0_average = (R0_2 + R0_3) / 2 # We averaged the last two R0 calculations since the infectious period is between 7-11 days. We hoped this would give us a better estimate of our R0 value.

print(f"Estimated R0 with infectious rate: {R0_average}") # Print the value to the terminal so that we can retrieve the data.

# %%

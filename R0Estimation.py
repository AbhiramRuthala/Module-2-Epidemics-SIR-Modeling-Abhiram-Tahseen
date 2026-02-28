#%%
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

#plot line graph i think
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['active reported daily cases'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.title('Active Cases Over Time')
plt.grid()
plt.tight_layout()
plt.show()

# Put fit to the exponential growth in the above graph
x = np.array(data['date'].map(pd.Timestamp.timestamp)).reshape(-1, 1)
y = np.array(data['active reported daily cases'])

logistics = np.polyfit(x.flatten(), y, 1)
m = logistics[0] # Slope of the graph
b = logistics[1]

x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = m * x_line + b

plt.plot(x, y, label='Fitted Line')
plt.plot(x_line, y_line, color='red', label='Exponential Fit')
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.title('Active Cases Over Time')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

#Estimations
infectious_Rate1 = 1/2
infectious_Rate2 = 1/7
infectious_Rate3 = 1/11
infectious_Rate4 = 1/9

beta_rate = m

R0_1 = beta_rate / infectious_Rate1
R0_2 = beta_rate / infectious_Rate2
R0_3 = beta_rate / infectious_Rate3

# R0_4 = beta_rate / infectious_Rate4
# R0_5 = beta_rate / 5

#Only do an average of the last two for an accurate R0
R0_average = (R0_2 + R0_3) / 2

print(f"Estimated R0 with infectious rate: {R0_average}")


# %%

#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv("C://Users//tta20//OneDrive - University of Virginia//BME 2315 (Comp)//Module 2//Module-2-Epidemics-SIR-Modeling-Abhiram-Tahseen//Data//mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time
data.plot.scatter(x='date', y='active reported daily cases')

plt.xlabel('Time (Dates)') 
plt.ylabel('Active Reported Cases') 
plt.title('Active Reported Cases vs Time')
plt.show()



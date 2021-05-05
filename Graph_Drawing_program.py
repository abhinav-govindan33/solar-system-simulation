# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:07:00 2021

@author: Abhinav
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from scipy import integrate 
import astropy.units as u 
import astropy.constants as c 

t_end = 220

data_JV_all = pd.read_csv("Diff_Vectors_for_JV_data.csv", sep=r'\s*,\s*', engine = 'python')
data_JV_AbN = pd.read_csv("Diff_Vectors_for_JV_data_AbN.csv", sep=r'\s*,\s*', engine = 'python')
data_JV_SUJSN = pd.read_csv("Diff_Vectors_for_JV_data_SUJSN.csv", sep=r'\s*,\s*', engine = 'python')
data_JV_SUJS = pd.read_csv("Diff_Vectors_for_JV_data_SUJS.csv", sep=r'\s*,\s*', engine = 'python')
data_JV_SUJ = pd.read_csv("Diff_Vectors_for_JV_data_SUJ.csv", sep=r'\s*,\s*', engine = 'python')
data_JV_SU = pd.read_csv("Diff_Vectors_for_JV_data_SU.csv", sep=r'\s*,\s*', engine = 'python')


diff_vectors_all = data_JV_all['Percent_error']
diff_vectors_AbN = data_JV_AbN['Percent_error']
diff_vectors_SUJSN = data_JV_SUJSN['Percent_error']
diff_vectors_SUJS = data_JV_SUJS['Percent_error']
diff_vectors_SUJ = data_JV_SUJ['Percent_error']
diff_vectors_SU = data_JV_SU['Percent_error']

time_ax = range(0,int(365.2422*t_end) + 1)
time = list(np.array(time_ax)/365)



plt.figure(1)
plt.title("Percentage Error vs Time")
plt.plot(time, diff_vectors_all, color = 'black', label = 'All')
plt.plot(time, diff_vectors_AbN, color = 'red', label = 'AbN')
plt.plot(time, diff_vectors_SUJSN, color = 'brown', label = 'SUJSN')
plt.plot(time, diff_vectors_SUJS, color = 'green', label = 'SUJS')
plt.plot(time, diff_vectors_SUJ, color = 'blue', label = 'SUJ')
plt.plot(time, diff_vectors_SU, color = 'violet', label = 'SU')
plt.xlabel('Time (Years)')
plt.ylabel('% Error')
plt.yscale('log')
plt.legend()
plt.savefig("AG_Final_Graph.pdf", bbox_inches='tight')
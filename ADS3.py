# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:04:31 2023

@author: liyak
"""

#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster 
import errors as err

# Reading input files
def read_file(x):
    """
    This function reads the file from the given address and
    loads it into a dataframe.Then transposes the dataframe and returns both
    the first and transposed dataframes. It also sets the header for the
    transposed dataframe
    Parameters
    ----------
    x : string
        Name of the file to be read into the dataframe.
    Returns
    -------
    A dataframe loaded from the file and it's transpose.
    """

    df = pd.read_csv(x)
    df_converted = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])

    df_converted = df_converted.replace(np.nan, 0)

    # Header
    df_converted = df_converted.rename(columns={'Country Name': 'Year'})
    df_transposed = np.transpose(df_converted)
    # Header setting
    header = df_transposed.iloc[0].values.tolist()
    df_transposed.columns = header
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={"index": "year"})
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.dropna()
    df_transposed["year"] = df_transposed["year"].str[:4]
    df_transposed["year"] = pd.to_numeric(df_transposed["year"])
    df_transposed["Ireland"] = pd.to_numeric(df_transposed["Ireland"])
    df_transposed["Japan"] = pd.to_numeric(df_transposed["Japan"])
    df_transposed["year"] = df_transposed["year"].astype(int)
    print(df_transposed['year'])
    return df, df_transposed


df_CO2,df_CO2trans = read_file("CO2 gaseous_fuel.csv")
df_pop,df_poptrans = read_file("Population growth.csv")
df_GDP,df_GDPtrans = read_file("GDP per capita.csv")

def curve_fun(t, scale, growth):
  """

  Parameters
  ----------
  t : TYPE
  List of values
  scale : TYPE
  Scale of curve.
  growth : TYPE
  Growth of the curve.
  Returns
  -------
  c : TYPE
  Result
  """
  c = scale * np.exp(growth * (t-1960))
  return c

#Call the file read function
df_CO2 = read_file("CO2 gaseous_fuel.csv")
df_Pop = read_file("Population growth.csv")
df_GDP = read_file("GDP per capita.csv")

# curve fitting for CO2 Emission in Ireland

param, cov = opt.curve_fit(curve_fun,df_CO2trans["year"],df_CO2trans["Ireland"],p0=[4e8,0.1])
sigma = np.sqrt(np.diag(cov))

# Error
low,up = err.err_ranges(df_CO2trans["year"],curve_fun,param,sigma)
df_CO2trans["fit_value"] = curve_fun(df_CO2trans["year"], * param)

#1.Plotting the C02 emission values for Ireland
plt.figure()
plt.title("CO2 emissions from gaseous fuel consumption (% of total) -Ireland")
plt.plot(df_CO2trans["year"],df_CO2trans["Ireland"],label="data")
plt.plot(df_CO2trans["year"],df_CO2trans["fit_value"],c="red",label="fit")
plt.fill_between(df_CO2trans["year"],low,up,alpha=0.1)
plt.legend()
plt.xlim(1980,2015)
plt.xlabel("Year")
plt.ylabel("CO2 emission(%)")
plt.savefig("CO2 Emission_Ireland.png", dpi = 300, bbox_inches='tight')
plt.show()

#2.Plotting the predicted values for CO2 Emission in Ireland 
plt.figure()
plt.title("Prediction of Ireland's CO2 emission from gaseous fuel consumption for 2035")
pred_yr = np.arange(1980,2035)
pred_irl = curve_fun(pred_yr,*param)
plt.plot(df_CO2trans["year"],df_CO2trans["Ireland"],label="data")
plt.plot(pred_yr,pred_irl,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2 Emission(%)")
plt.savefig("CO2_Emission_Ireland_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()


#Curve fitting for Japan
param, cov = opt.curve_fit(curve_fun,df_CO2trans["year"],df_CO2trans["Japan"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
# Error
low,up = err.err_ranges(df_CO2trans["year"],curve_fun,param,sigma)
df_CO2trans["fit_value"] = curve_fun(df_CO2trans["year"], * param)

#3.Plotting the predicted values for CO2 emission in Japan
plt.figure()
plt.title("Japan'S CO2 emission from gaseous fuel consumption prediction For 2035")
pred_yr = np.arange(1980,2035)
pred_jpn = curve_fun(pred_yr,*param)
plt.plot(df_CO2trans["year"],df_CO2trans["Japan"],label="data")
plt.plot(pred_yr,pred_jpn,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2 emission(%)")
plt.savefig("CO2_Japan_Prediction.png", dpi = 300, bbox_inches='tight')
plt.show()

#curve fitting for Ireland
param, cov = opt.curve_fit(curve_fun,df_poptrans["year"],df_poptrans["Ireland"],p0=[4e8,0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
#Error
low,up = err.err_ranges(df_poptrans["year"],curve_fun,param,sigma)
df_poptrans["fit_value"] = curve_fun(df_poptrans["year"], * param)


#4.Plotting the annual Population growth of Ireland

plt.figure()
plt.title("Population growth (annual %) - Ireland")
plt.plot(df_poptrans["year"],df_poptrans["Ireland"],label="data")
plt.plot(df_poptrans["year"],df_poptrans["fit_value"],c="maroon",label="fit")
plt.fill_between(df_poptrans["year"],low,up,alpha=0.5)
plt.legend()
plt.xlim(1980,2020)
plt.xlabel("Year")
plt.ylabel("Population growth (annual %))")
plt.savefig("Population_growth_Japan.png", dpi = 300, bbox_inches='tight')
plt.show()

# Curve fit for Population growth Japan
param, cov = opt.curve_fit(curve_fun,df_poptrans["year"],df_poptrans["Japan"],p0=[4e8,0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
# Error
low,up = err.err_ranges(df_poptrans["year"],curve_fun,param,sigma)
df_poptrans["fit_value"] = curve_fun(df_poptrans["year"], *param)

#5.Plotting the Population growth prediction of Japan
plt.figure()
plt.title("Japan Population growth prediction for 2035")
pred_yr = np.arange(1980,2035)
pred_jpn = curve_fun(pred_yr,*param)
plt.plot(df_poptrans["year"],df_poptrans["Japan"],label="data")
plt.plot(pred_yr,pred_jpn,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Population growth (annual %)")
plt.savefig("Population_growth_Prediction_Japan.png", dpi = 300, bbox_inches='tight')
plt.show()

#6.Plotting GDP per capita of Ireland & Japan
print(df_GDPtrans)
plt.figure()
plt.plot(df_GDPtrans["year"], df_GDPtrans["Japan"])
plt.plot(df_GDPtrans["year"], df_GDPtrans["Ireland"])
plt.xlim(1980,2020)
plt.xlabel("Year")
plt.ylabel("GDP per capita Growth")
plt.legend(['JPN','IRL'])
plt.title("GDP per capita growth (annual %)")
plt.savefig("GDP_per_capita_growth.png", dpi = 300, bbox_inches='tight')
plt.show()

df_CO2trans= df_CO2trans.iloc[:, [112,120]]
print(df_CO2trans)

kmean = cluster.KMeans(n_clusters=2).fit(df_CO2trans)
label = kmean.labels_
print(label)

#7.Scatter plot  for Japan & Ireland CO2 Emission
plt.scatter(df_CO2trans["Ireland"], df_CO2trans["Japan"], c=label, cmap="rainbow")
plt.title("Ireland and Japan - CO2 Emission")
c = kmean.cluster_centers_
plt.savefig("Scatter_Japan_Ireland_CO2.png", dpi = 300, bbox_inches='tight')
plt.show()

ireland = pd.DataFrame()
#print(ireland)
ireland["CO2_emission_gaseous_fuel"] = df_CO2trans["Ireland"]
ireland["Population_growth"] = df_poptrans["Ireland"]

#col = np.array(ireland["CO2_emission_gaseous_fuel"]).reshape(-1,1)
kmean = cluster.KMeans(n_clusters=2).fit(ireland)
label = kmean.labels_

#8.Scatter plot CO2 v/s Population growth use of ireland
plt.scatter(ireland["CO2_emission_gaseous_fuel"], ireland["Population_growth"], c=label,cmap="jet")
plt.title("CO2 emission vs Population growth - Ireland")
plt.savefig("Scatter_CO2_vs_Population_Ireland.png", dpi = 300, bbox_inches='tight')
c = kmean.cluster_centers_
print("centers",c)

for t in range(2):
  xc,yc = c[t,:]
  print("xc",xc)
  plt.plot(xc,yc,"ok",markersize=8)
  plt.figure()
  plt.savefig("Scatter_CO2_vs_Population_Ireland.png", dpi = 300, bbox_inches='tight')
  plt.show()
  

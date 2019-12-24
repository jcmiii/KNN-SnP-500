"""
Program: snp500_knn.py
Date created: 6/20/2017
Last modified: 8/31/2017

Description: this program implements K Nearest Neighbors to predict large
movements in the S&P 500.
"""
##############################################################################
# Import libraries                                                           #
##############################################################################

# import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# print (sklearn.__version__)
import myFcns
#import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D
# import sklearn as skl
# from matplotlib.colors import ListedColormap

##############################################################################
# Parameters & Constants - change as needed                                  #
##############################################################################
# Turn off/on a warning related to working with slices of dataframes
pd.options.mode.chained_assignment = None   

# Data related. Note that most recent data is at bottom of file
dataFile = 'SnP500features_edited.csv'      # Path to data file
trnCutDate = pd.to_datetime('2005-08-31')   # Cut dates for training, testing 
proCutDate = pd.to_datetime('2006-08-31')   #            and production data. 

# Testing/training parameters
show_pro = True                   # Hide production results when testing
plot_rtns = False                 # Show plot of returns?
rollingMean = False               # Calculate rolling mean for closing prices?
daysRollingMean = 2               # Days in rolling mean calc - for smoothing
mo4_days = 25                     # longest momentum lookback
numNhbrs = 11                     # No. neighbors to use in K-NN
numComponents = 5                 # No. components to keep after PCA
tstPct = 0.09                     # Pct of data to use for test data
numEpochs = 70                    # No. of training cycles to test
fwdN = 60                         # Window (in days) for calculating forward
                                  #                 return (e.g. 5 = 5 days)

# Thresholds for bull, neutral, bear on forward returns. 0.01 = 1%
longThreshold = 0.012
shortThreshold = -0.015

# Cols to use for training.
trn_cols = ['NHNL','D1_NHNL','D2_NHNL','D3_NHNL','D4_NHNL','Last96dHi','Bull', 
            'Bear','Neutral','close/96dHi','Volatility','Mo1','Mo2','Mo3',
            'Mo4']  #'10yrT_yld', 'Cos', 'Sin'
##############################################################################
# Read data into data frame                                                  #
##############################################################################
# Read data from csv file. Note that most recent data is at bottom of file
snpDF = pd.read_csv(dataFile)
print('(rows, cols) = ', snpDF.shape)

##############################################################################
# Convert 'Date' field from str to pandas.tslib.Timestamp                    #
##############################################################################
snpDF['Date'] = pd.to_datetime(snpDF['Date'])
# print(type(snpDF['Date'][1]))

##############################################################################
# 5. Add predictive features                                                 #
##############################################################################
# Volatility feature: (day hi - day lo)/close
snpDF['Volatility'] = (snpDF['Hi'] - snpDF['Lo']) / snpDF['Close']

# Momentum features
# 1st smooth
smoothing_days = 2
closing = snpDF['Close'].rolling(smoothing_days).mean()
# Fill in first few NaN with the original prices
closing[ : smoothing_days - 1] = snpDF['Close'][ : smoothing_days - 1]
# Calculate momentum
snpDF['Mo1'] = np.roll(closing,  5) / closing       # 1 week
snpDF['Mo2'] = np.roll(closing, 10) / closing       # 2 week
snpDF['Mo3'] = np.roll(closing, 15) / closing       # 3 week
snpDF['Mo4'] = np.roll(closing, mo4_days) / closing       # 4 week

# Change in New Hi - New Lo. First few rows will not have a 
# valid value. These rows are removed below.
snpDF['D1_NHNL'] = snpDF['NHNL'] - np.roll(snpDF['NHNL'], 1)
snpDF['D2_NHNL'] = snpDF['NHNL'] - np.roll(snpDF['NHNL'], 2)
snpDF['D3_NHNL'] = snpDF['NHNL'] - np.roll(snpDF['NHNL'], 3)
snpDF['D4_NHNL'] = snpDF['NHNL'] - np.roll(snpDF['NHNL'], 4)

# Seasonality feature. Based on 252 trading days per year: 2*Pi/252 = Pi/126
snpDF['Cos'] = np.cos( (math.pi / 126) * snpDF.index)
snpDF['Sin'] = np.sin( (math.pi / 126) * snpDF.index)
##############################################################################
# Add daily returns                                                          #
##############################################################################
# Tomorrow's close / Today's close. Represents the return you would get if
# you endered the market at close based on a signal prior to close and then
# stayed in the market for one day. It is really the next day's mkt rtn, but
# placing it with today's data will make calculating returns easier, since it
# will coincide with the signal.
#
#Note: most recent data is at bottom of dataframe
snpDF['DailyRtn'] = np.roll(snpDF['Close'], -1) / snpDF['Close']  

# Note: last row does not have a rtn (no Tomorrow's Close). It will be dropped
# below.

##############################################################################
# 6. Add forward returns and output (bull, bear, neutral indicator)          #
##############################################################################
# Smooth closing prices 
if rollingMean:
    closing = snpDF['Close'].rolling(daysRollingMean).mean()
    # Fill in first few NaN with the original prices
    closing[:daysRollingMean - 1] = snpDF['Close'][:daysRollingMean - 1]
else: 
    closing = snpDF['Close']

# Today's close / Close fwdN days forward. Represents mkt return over next
# fwdN days. Note that most recent data is at bottom of dataframe, so bottom
# fwdN entries will not have valid forward returns. These are removed below
snpDF['FwdRtn'] = (np.roll(closing, -fwdN) / closing) - 1 

# Classify Rtn into 1 of 3 categories: bull=1, bear=-1, neutral=0
snpDF['RtnCat'] = snpDF['FwdRtn'].apply(lambda x: 1 if x > longThreshold else 
                  (-1 if x < shortThreshold else 0) )

##############################################################################
# Drop rows that do not have valid data                                      #
##############################################################################
# Drop last fwdN rows (no fwd return), and 1st rows w/ NaN (no momentum) 
# See sections 5 and 6 above.
snpDF = snpDF[mo4_days : -fwdN]
#print(snpDF[:2])

##############################################################################
# Create training, testing, and production data sets                         #
##############################################################################
print('snpDF headings: ')
print(list(snpDF))

# Select rows prior to specified date for training and testing
trnTstDF = snpDF.loc[snpDF['Date'] <= trnCutDate]
# Select rows between two dates for production (original)
proDF_org = snpDF.loc[(snpDF['Date'] > trnCutDate) & 
                      (snpDF['Date'] <= proCutDate)]

##############################################################################
# Run randomized tests                                                       #
##############################################################################
# Initialize arrays to hold forecast results
trn_ratios = np.zeros( (numEpochs, 4, 4) )
tst_ratios = np.zeros( (numEpochs, 4, 4) )
pro_ratios = np.zeros( (numEpochs, 4, 4) )
pro_rtns = np.zeros( (numEpochs, 3) )
for i in range(numEpochs):
    
    # Split into training and testing (from sklearn library)
    trnDF, tstDF = train_test_split(trnTstDF, test_size = tstPct)
    # Initialize proDF as a copy of original
    proDF = proDF_org.copy(deep=True)

    ##########################################################################
    # Get stats from training data                                           #
    ##########################################################################
    # getStats returns a dataframe with the min, max, mean, st dev of each col
    statsDF = myFcns.getStats(trnDF)

    ##########################################################################
    # Scale columns for trn, tst, pro data files                             #
    ##########################################################################
    # Scale some cols into [a,b]
    a = 0
    b = 1
    cols_to_scale = ['NHNL','D1_NHNL','D2_NHNL','D3_NHNL','D4_NHNL','Bull', 
                     'Bear','Neutral','close/96dHi','10yrT_yld','Volatility', 
                     'Mo1','Mo2','Mo3','Mo4','Cos','Sin'] 
    myFcns.lin_scale(trnDF, cols_to_scale, statsDF, a, b)
    myFcns.lin_scale(tstDF, cols_to_scale, statsDF, a, b)
    myFcns.lin_scale(proDF, cols_to_scale, statsDF, a, b)
    
    # Scale days since last 96 day hi using log 
    trnDF.Last96dHi = np.log10(trnDF.Last96dHi + 1)
    tstDF.Last96dHi = np.log10(tstDF.Last96dHi + 1)
    proDF.Last96dHi = np.log10(proDF.Last96dHi + 1)
    ##########################################################################
    # Perform PCA                                                            #
    ##########################################################################
    # Import
    from sklearn.decomposition import PCA
  
    # Create PCA object
    skl_pca = PCA(n_components = numComponents)
    
    # Perform PCA on desired cols of trnDF 
    skl_pca.fit(trnDF[trn_cols])
    
    # Print explained variance
    #print('PCA explained variance %: ', skl_pca.explained_variance_ratio_)
    
    # Transform training, test, and production data
    pca_trn = skl_pca.transform(trnDF[trn_cols])
    pca_tst = skl_pca.transform(tstDF[trn_cols])
    pca_pro = skl_pca.transform(proDF[trn_cols])    
    
    ##########################################################################
    # Run K-NN                                                               #
    ##########################################################################
    # Import the K-NN solver
    from sklearn import neighbors
    
    # create classifier object. p = exponent for p-norm.
    clf = neighbors.KNeighborsClassifier(n_neighbors = numNhbrs, p = 2)#, 
                                         #weights = 'distance')
    
    # Fit the model on training data
    clf.fit(pca_trn, trnDF.RtnCat)
    
    # Get predictions (numpy arrays). Store in DF
    trnDF['Forecast'] = clf.predict(pca_trn)
    tstDF['Forecast'] = clf.predict(pca_tst)
    proDF['Forecast'] = clf.predict(pca_pro)

    # Store accuracy in 3D arrays. 
    # For each i (epoch) the 2D array has format:
    #        
    #                            Actual          Forecast
    #                    Bear   Neutral   Bull   Sum(Days) 
    #               Bear  %        %       X       int
    # Forecast   Neutral  %        %       %       int
    #               Bull  %        %       %       int
    #   Actual Sum(Days) int      int     int      
    # 
    # E.g. X = N( (Bear forecast) && (Bull actual) ) / N(Bear forecast)
    for j in range(3):
        # Calcluate actual number in each category(j=0 -> Bear, etc.)
        # Note: production is the same for each iteration. Could be taken
        # out of the loop and calculated once below.
        trn_ratios[i,3,j] = trnDF.RtnCat[trnDF.RtnCat == j-1].count() 
        tst_ratios[i,3,j] = tstDF.RtnCat[tstDF.RtnCat == j-1].count() 
        pro_ratios[i,3,j] = proDF.RtnCat[proDF.RtnCat == j-1].count()
        
        # Calculate tot num forecast for category (j=0 -> Bear, etc.)
        trn_ratios[i,j,3] = trnDF.RtnCat[trnDF.Forecast == j-1].count() 
        tst_ratios[i,j,3] = tstDF.RtnCat[tstDF.Forecast == j-1].count() 
        pro_ratios[i,j,3] = proDF.RtnCat[proDF.Forecast == j-1].count()

        # Calculate production returns
        pro_rtns[i,j] = proDF.DailyRtn[proDF.Forecast == j-1].product()

        for k in range(3):
            # i = epoch
            # j-1 = forecast: bear=-1, neutral=0, bull=1
            # k-1 = actual
            # Training data results and ratios
            num_forecast_in_cat = trnDF.RtnCat[(trnDF.Forecast == j-1) & 
                             (trnDF.RtnCat == k-1)].count() 
            if trn_ratios[i,j,3] == 0:
                trn_ratios[i,j,k] = 0
            else:
                # trn_ratios[i,j,3] = Tot num forecast 
                trn_ratios[i,j,k] = num_forecast_in_cat / trn_ratios[i,j,3]

            # Testing data results and ratios
            num_forecast_in_cat = tstDF.RtnCat[(tstDF.Forecast == j-1) & 
                             (tstDF.RtnCat == k-1)].count() 
            if tst_ratios[i,j,3] == 0:
                tst_ratios[i,j,k] = 0
            else:
                # trn_ratios[i,j,3] = Tot num forecast 
                tst_ratios[i,j,k] = num_forecast_in_cat / tst_ratios[i,j,3] 
            
            # Production data results and ratios
            num_forecast_in_cat = proDF.RtnCat[(proDF.Forecast == j-1) & 
                                               (proDF.RtnCat == k-1)].count() 
            if pro_ratios[i,j,3] == 0: 
                pro_ratios[i,j,k] = 0
            else:
                # trn_ratios[i,j,3] = Tot num forecast 
                pro_ratios[i,j,k] = num_forecast_in_cat / pro_ratios[i,j,3]

##############################################################################
# Calculate average percents over all runs                                   #
##############################################################################
# Columns and Indexes for dataframes
cols = ['Bear', 'Neutral', 'Bull', 'ForecastDays']
idxs = ['Bear', 'Neutral', 'Bull', 'ActualDays']

# Set up dataframes
trn_avgDF = pd.DataFrame(columns=cols, index=idxs)
tst_avgDF = pd.DataFrame(columns=cols, index=idxs)
pro_avgDF = pd.DataFrame(columns=cols, index=idxs)

# Calculate averages
forecast = 0
for idx in idxs:
    actual = 0
    for col in cols:
        trn_avgDF[col][idx] = np.mean(trn_ratios[:, forecast, actual])
        tst_avgDF[col][idx] = np.mean(tst_ratios[:, forecast, actual])
        pro_avgDF[col][idx] = np.mean(pro_ratios[:, forecast, actual])
        actual = actual + 1
    forecast = forecast + 1

# Calculate total days
trn_avgDF['ForecastDays']['ActualDays'] = trnDF.RtnCat.count()        
tst_avgDF['ForecastDays']['ActualDays'] = tstDF.RtnCat.count()        
pro_avgDF['ForecastDays']['ActualDays'] = proDF.RtnCat.count()        
    
# Print parameters
print()
print('Trn data from ', snpDF['Date'][mo4_days], ' to ', trnCutDate)
print('Pro data from ', trnCutDate, ' to ', proCutDate)
print('Tst data % : ', 100*tstPct)
print('Rolling Mean = ', rollingMean)
print('Days in rolling mean: ', daysRollingMean)
print('Num neighbors: ', numNhbrs)
print('Num PCA components: ', numComponents)
print('Num epochs: ', numEpochs)
print('Num days in fwd rtn: ', fwdN)
print('Bull threshold: ', longThreshold)
print('Bear threshold: ', shortThreshold)
print('Training features: ', trn_cols)

# Print averages
print()
print('Averages: row=prediction, col=actual')
print('Trn avg:')
print(trn_avgDF)
print()
print('Tst avg:')
print(tst_avgDF)
print()
if show_pro:
    print('Pro avg:')
    print(pro_avgDF)
    print()

# Print returns
if show_pro:
    print('Production Returns:')
    print('S&P 500 Return: ', proDF.DailyRtn.product())
    print('Bear K-NN Rtrn: ', np.mean(pro_rtns[:, 0]))
    print('Nutrl K-NN Rtn: ', np.mean(pro_rtns[:, 1]))
    print('Bull K-NN Rtrn: ', np.mean(pro_rtns[:, 2]))

# Plot returns from last epoch
if plot_rtns:
    # Returns for the S&P 500
    proDF['snpRtn'] = proDF.DailyRtn.cumprod()

    # Returns of K-NN when bullish
    proDF['BullRtn'] = proDF.DailyRtn[proDF['Forecast'] == 1].cumprod()
    # Replace days when not in market with previous days cumulative return
    # Check first entry for NaN
    if np.isnan(proDF.BullRtn.iloc[0]):
        proDF.BullRtn.iloc[0] = 1
    # Replaces NaN with previous non-NaN value
    proDF.BullRtn = proDF.BullRtn.ffill()   

    # Returns of K-NN when neutral
    proDF['NtrlRtn'] = proDF.DailyRtn[proDF['Forecast'] == 0].cumprod()
    # Replace days when not in market with previous days cumulative return
    # Check first entry for NaN
    if np.isnan(proDF.NtrlRtn.iloc[0]):
        proDF.NtrlRtn.iloc[0] = 1
    # Replaces NaN with previous non-NaN value
    proDF.NtrlRtn = proDF.NtrlRtn.ffill()   
    
    # Returns of K-NN when bearish
    proDF['BearRtn'] = proDF.DailyRtn[proDF['Forecast'] == -1].cumprod()
    # Replace days when not in market with previous days cumulative return
    # Check first entry for NaN
    if np.isnan(proDF.BearRtn.iloc[0]):
        proDF.BearRtn.iloc[0] = 1
    # Replaces NaN with previous non-NaN value
    proDF.BearRtn = proDF.BearRtn.ffill()   

    # Create plot
    ax = proDF.snpRtn.plot(c = 'grey')
    proDF.BullRtn.plot(c = 'green', ax = ax)
    proDF.NtrlRtn.plot(c = 'yellow', ax = ax)
    proDF.BearRtn.plot(c = 'red', ax = ax)
    
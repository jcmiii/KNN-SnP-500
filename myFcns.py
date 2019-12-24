##############################################################################
# Display Boxplots                                                           #
##############################################################################
def displayBoxplot(df, featuresList):
    # Import libraries
    import matplotlib.pylab as plt

    print("Showing Boxplots: Bull, Neutral, Bear")

    # boxplots for bullish data points
    df[df['RtnBin'] == 1].boxplot(featuresList)
    plt.show()
    # boxplots for neutral data points (in new window)
    plt.figure()
    df[df['RtnBin'] == 0].boxplot(featuresList)
    plt.show()
    # boxplots for bearish data points (in new window)
    plt.figure()
    df[df['RtnBin'] == -1].boxplot(featuresList)
    plt.show()
##############################################################################
# Display 2D Scatterplot                                                     #
##############################################################################
def display2DScatterplot(df, fx, fy, size):
    # Create neutral scatter plot and save it
    ax = df[df['RtnBin'] == 0].plot.scatter(x = fx, y = fy, 
             c = 'yellow', s = size)
    # Create bull scatter plot and save it on top of 1st
    df[df['RtnBin'] == 1].plot.scatter(x = fx, y = fy, 
              c = 'green', s = size, ax = ax)
    # Create bear scatter plot and save it on top of 1st
    df[df['RtnBin'] == -1].plot.scatter(x = fx, y = fy, 
             c = 'red', s = size, ax = ax)
##############################################################################
# Display 3D Scatterplot                                                     #
##############################################################################
def display3DScatterplot(df, fx, fy, fz, size):
    # Import libraries
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Set up 3D axes
    #plot3D = plt.figure().gca(projection='3d')
    plot3D = plt.figure()
    ax = Axes3D(plot3D)
    # Neutral
    ax.scatter(df[df['RtnBin'] == 0][fx], 
               df[df['RtnBin'] == 0][fy], 
               df[df['RtnBin'] == 0][fz], 
               c = 'yellow', s = size)
    # Bull
    ax.scatter(df[df['RtnBin'] == 1][fx], 
               df[df['RtnBin'] == 1][fy], 
               df[df['RtnBin'] == 1][fz], 
               c = 'green', s = size)
    # Bear
    ax.scatter(df[df['RtnBin'] == -1][fx], 
               df[df['RtnBin'] == -1][fy], 
               df[df['RtnBin'] == -1][fz], 
               c = 'red', s = size)
    plt.show()
##############################################################################
# Display PCA Scatterplot                                                    #
##############################################################################
def PCAscatterplot(pca_coords, df):
    import matplotlib.pylab as plt
    plt.plot(pca_coords[df.RtnBin== 0,0],pca_coords[df.RtnBin== 0,1],'y.')
    plt.plot(pca_coords[df.RtnBin== 1,0],pca_coords[df.RtnBin== 1,1],'g.')
    plt.plot(pca_coords[df.RtnBin==-1,0],pca_coords[df.RtnBin==-1,1],'r.')
    plt.show()
    
##############################################################################
# Get Stats                                                                  #
##############################################################################
def getStats(df):    
    # Calculate stats from dataframe.
    # Return max, min, mean, standard deviation in a dataframe
    
    # Import libraries
    import pandas as pd
    
    # Finds stats for each col, saves result in pandas series
    dfMin = df.min()
    dfMax = df.max()
    dfMean = df.mean()
    dfStd = df.std()

    # Name each pandas series
    dfMin.name = 'min'
    dfMax.name = 'max'
    dfMean.name = 'mean'
    dfStd.name = 'std'

    # Concatinate pandas series to create a dataFrame. 
    # axis=1 means col, not row
    statsDF = pd.concat((dfMin, dfMax, dfMean, dfStd), axis=1)
    statsDF = statsDF.T
    return statsDF
##############################################################################
# Linear scale into [a,b]                                                    #
##############################################################################
def lin_scale(df, scale_cols, statsDF, a, b):    
    # Scale col into [a,b]
    for x in scale_cols:
        df[x] = a + (b - a) * ( (df[x] - statsDF[x]['min']) / 
                  (statsDF[x]['max'] - statsDF[x]['min']) ) 
    

    
    
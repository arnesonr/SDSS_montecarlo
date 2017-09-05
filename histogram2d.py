import numpy as n

def twodhist(b, g, x_bins, y_bins):
    """
    PURPOSE: Generates a 2d histogram for use with imshow()
    
    USAGE:  array = 2dhist(array, array, int, int)
    
    ARGUMENTS:
    
       b: Einstein radius array in the x-coordinate of histogram
       g: power-law index (gamma) array in the y-coordinate of histogram
       x_bins: integer number of bins in x coordinate of histogram
       y_bins: integer number of bins in y coordinate of histogram

    RETURNS:  2d array for plotting using imshow()
    
    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    #determine delta_x and delta_y of histogram
    #x range will always be (0,5) and y range (0,2)
    dx = 5.0/x_bins
    dy = 2.0/y_bins
    how_many = n.size(b)
    A=list()
    
    for y in range(0,y_bins):
        row = n.zeros(x_bins)
        for x in range(0,x_bins):
            counter = 0.0
            for i in range(0,how_many):
                if b[i] <= (dx + dx*x) and b[i] > (dx + dx*(x-1)) and g[i] <= (2.0 - dy*y) and g[i] > (2.0 - dy*(y+1)):
                    counter = counter + 1
            row[x] = counter
        A.append(row)
    #normalize A
    A = A-n.min(A)
    A=A/n.max(A)
    return A

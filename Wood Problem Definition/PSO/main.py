#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:18:31 2020

@author: gtsal
"""

from WoodProblemDefinition import Stock, Order1, Order2, Order3
from shapely.geometry import Point
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
import math
import shapely
from descartes import PolygonPatch
from shapely.ops import cascaded_union
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

from DynNeighborPSO import DynNeighborPSO

# Simple helper class for getting matplotlib patches from shapely polygons with different face colors
class PlotPatchHelper:
    # a colormap with 41 colors
    CMapColors = np.array([
            [0,0.447,0.741,1],
            [0.85,0.325,0.098,1],
            [0.929,0.694,0.125,1],
            [0.494,0.184,0.556,1],
            [0.466,0.674,0.188,1],
            [0.301,0.745,0.933,1],
            [0.635,0.078,0.184,1],
            [0.333333333,0.333333333,0,1],
            [0.333333333,0.666666667,0,1],
            [0.666666667,0.333333333,0,1],
            [0.666666667,0.666666667,0,1],
            [1,0.333333333,0,1],
            [1,0.666666667,0,1],
            [0,0.333333333,0.5,1],
            [0,0.666666667,0.5,1],
            [0,1,0.5,1],
            [0.333333333,0,0.5,1],
            [0.333333333,0.333333333,0.5,1],
            [0.333333333,0.666666667,0.5,1],
            [0.333333333,1,0.5,1],
            [0.666666667,0,0.5,1],
            [0.666666667,0.333333333,0.5,1],
            [0.666666667,0.666666667,0.5,1],
            [1,0,0.5,1],
            [1,0.333333333,0.5,1],
            [1,0.666666667,0.5,1],
            [1,1,0.5,1],
            [0,0.333333333,1,1],
            [0,0.666666667,1,1],
            [0,1,1,1],
            [0.333333333,0,1,1],
            [0.333333333,0.333333333,1,1],
            [0.333333333,0.666666667,1,1],
            [0.333333333,1,1,1],
            [0.666666667,0,1,1],
            [0.666666667,0.333333333,1,1],
            [0.666666667,0.666666667,1,1],
            [0.666666667,1,1,1],
            [1,0,1,1],
            [1,0.333333333,1,1],
            [1,0.666666667,1,1]
            ])
    
    
    # Alpha controls the opaqueness, Gamma how darker the edge line will be and LineWidth its weight
    def __init__(self, Gamma=1.3, Alpha=0.9, LineWidth=2.0):
        self.Counter = 0
        self.Gamma = Gamma          # darker edge color if Gamma>1 -> faceColor ** Gamma; use np.inf for black
        self.Alpha = Alpha          # opaqueness level (1-transparency)
        self.LineWidth = LineWidth  # edge weight
    
    # circles through the colormap and returns the FaceColor and the EdgeColor (as FaceColor^Gamma)
    def nextcolor(self):
        col = self.CMapColors[self.Counter,:].copy()
        self.Counter = (self.Counter+1) % self.CMapColors.shape[0]
        return (col, col**self.Gamma)
    
    # returns a list of matplotlib.patches.PathPatch from the provided shapely polygons, using descartes; a list is 
    # returned even for a single polygon for common handling
    def get_patches(self, poly):
        if not isinstance(poly, list): # single polygon, make it a one element list for common handling
            poly = [poly]
        patchList = []
        for p in poly:
            fCol, eCol = self.nextcolor()
            patchList.append(PolygonPatch(p, alpha=self.Alpha, FaceColor=fCol, EdgeColor=eCol, 
                                          LineWidth=self.LineWidth))        
        return patchList


# Plots one or more shapely polygons in the provided axes ax. The named parameter values **kwargs are passed into
# PlotPatchHelper's constructor, e.g. you can write plotShapelyPoly(ax, poly, LineWidth=3, Alpha=1.0). Returns a list
# with the drawn patches objects even for a single polygon, for common handling
def plotShapelyPoly(ax, poly, **kwargs):
    return [ax.add_patch(p) for p in PlotPatchHelper(**kwargs).get_patches(poly)]



def ObjectiveFcnNew(particle,nVars,Stock,Order):
#    res=0
#    newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),particle[j*3+2], origin='centroid') for j in range(len(Order))] 
#    unionNewOrder=shapely.ops.cascaded_union(newOrder)
#    difUnionNewOrder=unionNewOrder.difference(Stock) # take newOrder out of stock - inverse of remaining
#    
#    existOverlap = 0
#    areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
#    difArea = areaSum-unionNewOrder.area
#    
#    dist_from_zero = sum([newOrder[i].area*(newOrder[i].centroid.y)+newOrder[i].centroid.x for i in range(0,len(newOrder))])
#    
#    existOverlap = round(existOverlap,5)
#    dist_from_zero= round(dist_from_zero,6)
#    
#    res = difUnionNewOrder.area*10000 + difArea* 10000 +dist_from_zero*10
#    return res
#    """ MATLAB's peaks function -> objective (fitness function) """
#
    res=0
    newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),particle[j*3+2], origin='centroid') for j in range(len(Order))]
    remaining = Stock  
    unionNewOrder=shapely.ops.cascaded_union(newOrder)
    remaining = Stock.difference(unionNewOrder)
    
    outOfStock=unionNewOrder.difference(Stock) # take newOrder out of stock - inverse of remaining
    areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
    overlapArea = areaSum-unionNewOrder.area
    
    dist_from_zero = sum([newOrder[i].area*(newOrder[i].centroid.x+newOrder[i].centroid.y) for i in range(0,len(newOrder))])
    ch= (remaining.convex_hull)
    lamda = (ch.area)/(remaining.area)-1
    alpha = 1.11
    fsm = 1/(1+alpha*lamda)
    dist_from_zero= round(dist_from_zero,6)

    res = outOfStock.area*10000 + overlapArea*10000  +dist_from_zero/10*10 + 100*fsm

    return res


class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DynNeighborPSO object and updates all elements in the figure.
        
        The figure has a top row of 1 subplots. This shows the best-so-far global finess value .
        The bottom row shows the global best-so-far solution achieved by the algorithm and the remaining current stock after placement.
    """
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution, swarm, global fitness line) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
         
        # figure
        self.fig = plt.figure()
        self.ax=[1,2,3]
        self.ax[0] = plt.subplot(211)
        
        self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
        self.lineBestFit, = self.ax[0].plot([], [])
        
        # auto-arrange subplots to avoid overlappings and show the plot
        # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
        
        self.ax[1] = plt.subplot(223)
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2] = plt.subplot(224)
        self.ax[2].set_title('Remaining after set difference')
        self.fig.tight_layout()

    
    def update(self, pso):
        """ Updates the figure in each iteration provided a PSODynNeighborPSO object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        
        if pso.Iteration == -1:
            xdata = np.arange(pso.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
       
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(pso.GlobalBestFitness))
        
        # because of title and particles positions changing, we cannot update specific artists only (the figure
        # background needs updating); redrawing the whole figure canvas is expensive but we have to

        newOrder= pso.newOrder
        remaining = pso.remaining
        # NOTE: the above operation is perhaps faster if we perform a cascade union first as below, check it on your code:
        #remaining = Stock[6].difference(shapely.ops.cascaded_union(newOrder))
        #self.fig2, ax = plt.subplots(ncols=2)
        #self.fig2.canvas.set_window_title('Stock[6] cutting Order3 (translated & rotated)')
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2].set_title('Remaining after set difference')
        pp = plotShapelyPoly(self.ax[1], [pso.Stock]+newOrder)
        pp[0].set_facecolor([1,1,1,1])
        plotShapelyPoly(self.ax[2], remaining)
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.ax[2].set_xlim(self.ax[1].get_xlim())
        self.ax[2].set_ylim(self.ax[1].get_ylim())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
       

def OutputFcn(pso, figObj):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        
        Always returns False (== don't stop the iterative process)
    """
    if pso.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.5f}'.format(pso.Iteration, pso.GlobalBestFitness))
    
    figObj.update(pso)
    
    return False
    

if __name__ == "__main__":
    # in case someone tries to run it from the command-line...
    plt.ion()
    #np.random.seed(1)
    
   
    orderList = [Order1, Order2, Order3] #all orders
    numPolygons=sum([len(Order1), len(Order2), len(Order3)])
    
    finalList=[] 
    finalListPerStock=[]
    notFittedList=[]

    orderN = len(orderList)
    remaining = Stock.copy()
    remainingN= len(remaining)
    counter=0
    polygonsFitted=0
    iterationsList=[]
    while (orderList):
        fitted = 0
        counter=counter+1
        currentOrder = orderList[0] # define current Order (it may be a part of order that was split)
        # save sum of areas of current order's parts
        currentOrderArea= sum([currentOrder[w].area for w in range(0,len(currentOrder))])
        #currentOrderArea = cascaded_union(currentOrder).area
        # save area of each remaining
        remainingsArea = np.array([remaining[k].area for k in range(0,len(remaining))])
        # [x,y,theta] for each part so 3* len(currentOrder)
        nVars = len(currentOrder) * 3
        # find which stocks have bigger area than the order's area calculated
        # and keep them as possible solutions
        # try to fit order in them starting by the smallest one
        bigEnough=(np.where(remainingsArea>currentOrderArea))[0]
        realIndexes = np.argsort(remainingsArea[bigEnough])
        bigEnough=bigEnough[realIndexes]
        print(bigEnough)
        for stockIndex in bigEnough:
            print("Try Stockindex=%d   -> OrderIndex=%d"% (stockIndex,counter))
            # set currentStock for pso the stocks-remainings from the local list
            currentStock = remaining[stockIndex]
            # Set lower and upper bounds for the 3 variables for each particle
            # as the bounds of stocks
            (minx, miny, maxx, maxy) = currentStock.bounds
            LowerBounds = np.ones(nVars)
            w1= [b for b in range(0,nVars,3)]
            w2= [b for b in range(1,nVars,3)]
            w3= [b for b in range(2,nVars,3)]
            LowerBounds[w1]= minx 
            LowerBounds[w2]= miny
            LowerBounds[w3]= 0
            
            UpperBounds = np.ones(nVars)
            UpperBounds[w1] = maxx
            UpperBounds[w2] = maxy
            UpperBounds[w3] = 90*4 # it can also work with 2 discrete values 0,90 in range {0,90}
            ##np.random.seed(13)
            
            figObj = FigureObjects(minx, maxx) # no need           
            outFun = lambda x: OutputFcn(x, figObj)
            pso = DynNeighborPSO(ObjectiveFcnNew, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                         OutputFcn=outFun, UseParallel=False, MaxStallIterations=15,Stock=currentStock,Order=currentOrder,remaining=currentStock,newOrder=currentOrder)
    
            pso.optimize()
            # get result and temporary apply the transformations
            resultPositions = pso.GlobalBestPosition
            newOrder = [ shapely.affinity.rotate( 
                    shapely.affinity.translate(currentOrder[k], xoff=resultPositions[k*3], yoff=resultPositions[k*3+1]), 
                    resultPositions[k*3+2], origin='centroid') for k in range(len(currentOrder))]
            iterationsList.append(pso.Iteration)
            #if (xwrese )
            # find if the current order was placed inside the currentStock
            # if area of difference of currentStock from union of order is bigger than a tolerance
            # go to next choice
            unionNewOrder=shapely.ops.cascaded_union(newOrder)
            difUnionNewOrder=unionNewOrder.difference(currentStock) # take newOrder out of stock - inverse of remaining
            if difUnionNewOrder.area >0.0001:
                continue
            
            # check if there is overlap and skip
            # overlap area is equal with sumOfArea - areaOfUnion
            areaSum = sum([newOrder[w].area for w in range(0,len(newOrder))])
            difArea = areaSum-unionNewOrder.area
            if difArea > 0.0001:
                continue
            

            # this part of code is executed only if there is no overlap and no polygons out of stock and then it breaks the inner loop 
            fitted=1
            for p in newOrder:
                remaining[stockIndex] = remaining[stockIndex].difference(p)
            
            break
        #if polygons don't fit then split order in 2 parts
        if (fitted==0):
            #orderList.remove(currentOrder)
            if(int((len(currentOrder)/2))!=0):
                temp1=(currentOrder[0:int((len(currentOrder)/2))])
                temp2=(currentOrder[int((len(currentOrder)/2)):len(currentOrder)])
                orderList = [temp1]+[temp2] + orderList[1:]
            else:
                # if order contains only one polygon and cannot be fitted, it will add it to notFittedList and it will go to next order  
                notFittedList.append(orderList[0])
                orderList.remove(currentOrder)
                
        else:
            # if polygons of current order is fitted, then increase the number of fitted polygons, append the parts of order in finalList, append the stockIndex and remove the fitted order
            polygonsFitted=polygonsFitted+len(currentOrder)
            finalList.append( newOrder)
            finalListPerStock.append(stockIndex)
            orderList.remove(currentOrder)
            print("Fitted Stockindex=%d   -> OrderIndex=%d"% (stockIndex,counter))

    
    
    end = time.time()

    print("Polygons fitted=%d from %d polygon"%(polygonsFitted,numPolygons))
    print("\nNumber of Iterations (mean,min,max) = (%f,%f,%f)"%(np.mean(iterationsList),np.min(iterationsList),np.max(iterationsList)))
    # NOTE: the above operation is perhaps faster if we perform a cascade union first as below, check it on your code:
    #remaining = Stock[6].difference(shapely.ops.cascaded_union(newOrder))
    
    # Plot remainings
    ind=0
    fig, ax = plt.subplots(ncols=4,nrows=2, figsize=(16,9))
    fig.canvas.set_window_title('Remainings- Polygons fitted=%d from %d polygons'%(polygonsFitted,numPolygons))
    for i in range(0,len(Stock)):
        if i>=4:
            ind=1
        
        plotShapelyPoly(ax[ind][i%4], remaining[i])
        ax[ind][i%4].set_title('Remaining of Stock[%d]'%i)
        (minx, miny, maxx, maxy) = Stock[i].bounds
        ax[ind][i%4].set_ylim(bottom=miny,top=maxy)
        ax[ind][i%4].set_xlim(left=minx ,right=maxx)
        #ax[ind][i%4].relim()
        #ax[ind][i%4].autoscale_view()
    
    #Save figure with remainings
    import os 
    file_name = "PSOfull.png"
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".png")[0] + str(expand) + ".png"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
            
    fig.savefig(file_name)

    
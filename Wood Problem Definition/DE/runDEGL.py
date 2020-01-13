# -*- coding: utf-8 -*-
"""
Runs DynNeighborPSO for MATLAB's peaks 2D function.

"""

import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from Degl import Degl



def ObjectiveFcn(particle):
    """ MATLAB's peaks function -> objective (fitness function) """
    x = particle[0]
    y = particle[1]
    return 3.*(1-x)**2. * np.exp(-(x**2) - (y+1)**2) \
           - 10.*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
           - 1./3 * np.exp(-(x+1)**2 - y**2)



class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DynNeighborPSO object and updates all elements in the figure.
        
        The figure has a top row of two subplots. The left one is a 3D plot of the peaks function with only the global 
        best-so-far solution (red dot). The right one is the peaks function contour, together with the best-so-far 
        solution (red dot) and the positions of all particles in the current iteration's swarm (smaller black dots).
        The bottom row shows the best-so-far global finess value achieved by the algorithm.
    """
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution, swarm, global fitness line) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
        
        assert np.isscalar(LowerBound), "The input argument LowerBound must be scalar."
        assert np.isscalar(UpperBound), "The input argument LowerBound must be scalar."
        
        # create the peaks data
        nPoints = 100
        space = np.linspace(LowerBound, UpperBound, nPoints)
        xx, yy = np.meshgrid(space,space)
        
        # np.ravel gets the data as a 1D array (raw internal storage), so zz is a 1D arrray and then we reshape it 
        # to match xx & yy dimensions
        zz = np.array( [ObjectiveFcn([x,y]) for x,y in zip(np.ravel(xx), np.ravel(yy)) ] )
        zz = zz.reshape(xx.shape)
        
        # figure
        self.fig = plt.figure()
        
        # 3D axis: the peaks surface & global best point
        cmap = 'gist_earth'
        self.ax3DBest = self.fig.add_subplot(221, projection='3d', azim=-120)
        self.ax3DBest.plot_surface(xx, yy, zz, cmap=cmap)
        self.ax3DBest.set_xlim(LowerBound, UpperBound)
        self.ax3DBest.set_ylim(LowerBound, UpperBound)
        self.line3DBest, = self.ax3DBest.plot([np.nan], [np.nan], zs=[np.nan], zdir='z', linestyle='', 
                                                color='r', marker='o', markersize=4, markerfacecolor='r')
        
        # 2D axis: contour, current swarm positions, and globalbest (on top)
        self.ax2DBest = self.fig.add_subplot(222)
        self.countour2D = self.ax2DBest.contour(xx, yy, zz, levels=30, cmap=cmap, linewidths=1)
        self.line2DSwarm, = self.ax2DBest.plot([np.nan], [np.nan], linestyle='', 
                                               color='k', marker='o', markersize=2, markerfacecolor='k')
        self.line2DBest, = self.ax2DBest.plot([np.nan], [np.nan], linestyle='', 
                                                color='r', marker='o', markersize=4, markerfacecolor='r')
        self.ax2DBest.set_title(f'[{np.NaN},{np.NaN}]') # title is best-so-far position as [x,y]
        self.ax2DBest.grid()
        
        # global best fitness line
        self.axBestFit = plt.subplot(212)
        self.axBestFit.set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
        self.lineBestFit, = self.axBestFit.plot([], [])
        
        # auto-arrange subplots to avoid overlappings and show the plot
        self.fig.tight_layout()
    
    
    def update(self, pso):
        """ Updates the figure in each iteration provided a PSODynNeighborPSO object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        if pso.Iteration == -1:
            xdata = np.arange(pso.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        
        # update global best point in 3D plot
        self.line3DBest.set_xdata(pso.GlobalBestPosition[0])
        self.line3DBest.set_ydata(pso.GlobalBestPosition[1])
        self.line3DBest.set_3d_properties(pso.GlobalBestFitness)
        
        # update global best point in 2D plot
        self.line2DBest.set_xdata(pso.GlobalBestPosition[0])
        self.line2DBest.set_ydata(pso.GlobalBestPosition[1])
        self.ax2DBest.set_title('[{:.3f},{:.3f}]'.format(pso.GlobalBestPosition[0],pso.GlobalBestPosition[1]))
        
        # update current swarm's particels positions in 2D plot
        self.line2DSwarm.set_xdata(pso.Swarm[:,0])
        self.line2DSwarm.set_ydata(pso.Swarm[:,1])
        
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        self.axBestFit.relim()
        self.axBestFit.autoscale_view()
        self.axBestFit.title.set_text('Best-so-far global best fitness: {:g}'.format(pso.GlobalBestFitness))
        
        # because of title and particles positions changing, we cannot update specific artists only (the figure
        # background needs updating); redrawing the whole figure canvas is expensive but we have to
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
    """ Executed only when the file is run as a script. """
    
    # in case somebody tries to run it from the command line directly...
    plt.ion()
    
    # uncomment the following line to get the same results in each execution
#    np.random.seed(13)
    
    nVars = 2
    
    #peaks is defined typically defined from -3 to 3, but we set -5 to 5 here to make the problem a bit harder
    LowerBounds = -5.0 * np.ones(nVars)
    UpperBounds = 5.0 * np.ones(nVars)
    
    figObj = FigureObjects(LowerBounds[0], UpperBounds[0])
    
    # lambda functor (unnamed function) so that the output function appears to accept one argument only, the 
    # DynNeighborPSO object; behind the scenes, the local object figObj is stored within the lambda
    outFun = lambda x: OutputFcn(x, figObj)
    
    # UseParallel=True is actually slower for simple objective functions such as this, but may be useful for more 
    # demanding objective functions. Requires the joblib package to be installed.
    # MaxStallIterations=20 is the default. Check how the algorithms performs for larger MaxStallIterations 
    # (e.g., 100 or 200).
    pso = DynNeighborPSO(ObjectiveFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                         OutputFcn=outFun, UseParallel=False, MaxStallIterations=20)
    pso.optimize()
    
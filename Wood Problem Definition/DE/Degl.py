# -*- coding: utf-8 -*-
"""
Particle swarm minimization algorithm with dynamic random neighborhood topology.

The DynNeighborPSO class implements a (somewhat) simplified version of the particleswarm algorithm from MATLAB's 
Global Optimization Toolbox.
"""

import numpy as np
import warnings
import math

try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False



class Degl:
    """ Particle swarm minimization algorithm with dynamic random neighborhood topology.
        
        pso = DynNeighborPSO(ObjectiveFcn, nVars, ...) creates the DynNeighborPSO object stored in variable pso and 
            performs all swarm initialization tasks (including calling the output function once, if provided).
        
        pso.optimize() subsequently runs the whole iterative process.
        
        After initialization, the pso object has the following properties that can be queried (also during the 
            iterative process through the output function):
            o All the arguments passed during boject (e.g., pso.MaxIterations, pso.ObjectiveFcn,  pso.LowerBounds, 
                etc.). See the documentation of the __init__ member below for supported options and their defaults.
            o Iteration: the current iteration. Its value is -1 after initialization 0 or greater during the iterative
                process.
            o Swarm: the current iteration swarm (nParticles x nVars)
            o Velocity: the current velocity vectors (nParticles x nVars)
            o CurennetFitness: the current swarm's fitnesses for all particles (nParticles x 1)
            o PreviousBestPosition: the best-so-far positions found for each individual (nParticles x nVars)
            o PreviousBestFitness: the fitnesses of the best-so-far individuals (nParticles x 1)
            o GlobalBestFitness: the overall best fitness attained found from the beginning of the iterative process
            o GlobalBestPosition: the overall best position found from the beginning of the iterative process
            o AdaptiveNeighborhoodSize: the current neighborhood size
            o MinNeighborhoodSize: the minimum neighborhood size allowed
            o AdaptiveInertia: the current value of the inertia weight
            o StallCounter: the stall counter value (for updating inertia)
            o StopReason: string with the stopping reason (only available at the end, when the algorithm stops)
            o GlobalBestSoFarFitnesses: a numpy vector that stores the global best-so-far fitness in each iteration. 
                Its size is MaxIterations+1, with the first element (GlobalBestSoFarFitnesses[0]) reserved for the best
                fitness value of the initial swarm. Accordingly, pso.GlobalBestSoFarFitnesses[pso.Iteration+1] stores 
                the global best fitness at iteration pso.Iteration. Since the global best-so-far is updated only if 
                lower that the previously stored, this is a non-strictly decreasing function. It is initialized with 
                NaN values and therefore is useful for plotting it, as the ydata of the matplotlib line object (NaN 
                values are just not plotted). In the latter case, the xdata would have to be set to 
                np.arange(pso.MaxIterations+1)-1, so that the X axis starts from -1.
    """
    
    
    def __init__( self
                , ObjectiveFcn
                , nVars
                , LowerBounds = None
                , UpperBounds = None
                , D = None
                , MaxIterations = None
                , Nf = 0.1
                , a = 0.8
                , b = 0.8
                , w_min = 0.4
                , w_max = 0.6
                , FunctionTolerance = 1.0e-6
                , MaxStallIterations = 20
                , OutputFcn = None
                , UseParallel = False
                ):
        """ The object is initialized with two mandatory positional arguments:
                o ObjectiveFcn: function object that accepts a vector (the particle) and returns the scalar fitness 
                                value, i.e., FitnessValue = ObjectiveFcn(Particle)
                o nVars: the number of problem variables
            The algorithm tries to minimize the ObjectiveFcn.
            
            The arguments LowerBounds & UpperBounds lets you define the lower and upper bounds for each variable. They 
            must be either scalars or vectors/lists with nVars elements. If not provided, LowerBound is set to -1000 
            and UpperBound is set to 1000 for all variables. If vectors are provided and some of its elements are not 
            finite (NaN or +-Inf), those elements are also replaced with +-1000 respectively.
            
            The rest of the arguments are the algorithm's options:
                o SwarmSize (default:  min(100,10*nVars)): Number of particles in the swarm, an integer greater than 1.
                o SelfAdjustmentWeight (default: 1.49): Weighting (finite scalar) of each particle’s best position when
                    adjusting velocity.
                o SocialAdjustmentWeight (default: 1.49): Weighting (finite scalar) of the neighborhood’s best position 
                    when adjusting velocity.
                o InertiaRange (default: [0.1, 1.1]): Two-element real vector with same sign values in increasing 
                    order. Gives the lower and upper bound of the adaptive inertia. To obtain a constant (nonadaptive) 
                    inertia, set both elements of InertiaRange to the same value.
                o MinNeighborsFraction (default: 0.25): Minimum adaptive neighborhood size, a scalar in [0, 1].
                o FunctionTolerance (default: 1e-6): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o MaxIterations (default: 200*nVars): Maximum number of iterations.
                o MaxStallIterations (default: 20): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o OutputFcn (default: None): Output function, which is called at the end of each iteration with the 
                    iterative data and they can stop the solver. The output function must have the signature 
                    stop = fun(pso), returning True if the iterative process must be terminated. pso is the 
                    DynNeighborPSO object (self here). The output function is also called after swarm initialization 
                    (i.e., within this member function).
                o UseParallel (default: False): Compute objective function in parallel when True. The latter requires
                    package joblib to be installed (i.e., pip install joplib or conda install joblib).

        """
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        self.Nf = Nf
        self.a = a
        self.b = b
        self.w_min = w_min
        self.w_max = w_max
        
        
        
        
        # assert options validity (simple checks only) & store them in the object
        if D is None:
            self.D = min(200, 10*nVars)
        else:
            assert np.isscalar(D) and D > 1, \
                "The D option must be a scalar integer greater than 1 and not None."
            self.D = max(2, int(round(self.D)))
        
        
        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        if MaxIterations is None:
            self.MaxIterations = 100*nVars
        else:
            assert np.isscalar(MaxIterations), "The MaxIterations option must be a scalar integer greater than 0."
            self.MaxIterations = max(1, int(round(MaxIterations)))
        assert np.isscalar(MaxStallIterations), \
            "The MaxStallIterations option must be a scalar integer greater than 0."
        self.MaxStallIterations = max(1, int(round(MaxStallIterations)))
        
        self.OutputFcn = OutputFcn
        assert np.isscalar(UseParallel) and (isinstance(UseParallel,bool) or isinstance(UseParallel,np.bool_)), \
            "The UseParallel option must be a scalar boolean value."
        self.UseParallel = UseParallel
        
        # lower bounds
        if LowerBounds is None:
            self.LowerBounds = -1000.0 * np.ones(nVars)
        elif np.isscalar(LowerBounds):
            self.LowerBounds = LowerBounds * np.ones(nVars)
        else:
            self.LowerBounds = np.array(LowerBounds, dtype=float)
        self.LowerBounds[~np.isfinite(self.LowerBounds)] = -1000.0
        assert len(self.LowerBounds) == nVars, \
            "When providing a vector for LowerBounds its number of element must equal the number of problem variables."
        # upper bounds
        if UpperBounds is None:
            self.UpperBounds = 1000.0 * np.ones(nVars)
        elif np.isscalar(UpperBounds):
            self.UpperBounds = UpperBounds * np.ones(nVars)
        else:
            self.UpperBounds = np.array(UpperBounds, dtype=float)
        self.UpperBounds[~np.isfinite(self.UpperBounds)] = 1000.0
        assert len(self.UpperBounds) == nVars, \
            "When providing a vector for UpperBounds its number of element must equal the number of problem variables."
        
        assert np.all(self.LowerBounds <= self.UpperBounds), \
            "Upper bounds must be greater or equal to lower bounds for all variables."
        
        
        # check that we have joblib if UseParallel is True
        if self.UseParallel and not HaveJoblib:
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; swarm objective values will be computed in serial mode instead.""")
            self.UseParallel = False
        
        
        # Initial swarm: randomly in [lower,upper] and if any is +-Inf in [-1000, 1000]
        lbMatrix = np.tile(self.LowerBounds, (self.D, 1))
        ubMatrix = np.tile(self.UpperBounds, (self.D, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        self.z = lbMatrix + np.random.rand(self.D,nVars) * bRangeMatrix
        
        #find radius
        k = math.floor(self.D * self.Nf) # Neighborhood size
        
        
        
        # Initial velocity: random in [-(UpperBound-LowerBound), (UpperBound-LowerBound)]
        self.Velocity = -bRangeMatrix + 2.0 * np.random.rand(self.D,nVars) * bRangeMatrix
        
        # Initial fitness
        self.CurennetFitness = np.zeros(self.D)
        self.__evaluateDE()
        
        # Initial best-so-far individuals and global best
        self.PreviousBestPosition = self.z.copy()
        self.PreviousBestFitness = self.CurennetFitness.copy()
        
        bInd = self.CurennetFitness.argmin()
        self.GlobalBestFitness = self.CurennetFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
        
        # iteration counter starts at -1, meaning initial population
        self.Iteration = -1;
        
#        # Initial neighborhood & inertia
#        self.MinNeighborhoodSize = max(2, int(np.floor(nParticles * self.MinNeighborsFraction)));
#        self.AdaptiveNeighborhoodSize = self.MinNeighborhoodSize;
#        
#        if np.all(self.InertiaRange >= 0):
#            self.AdaptiveInertia = self.InertiaRange[1]
#        else:
#            self.AdaptiveInertia = self.InertiaRange[0]
        
        self.StallCounter = 0;
        
        # Keep the global best of each iteration as an array initialized with NaNs. First element is for initial swarm,
        # so it has self.MaxIterations+1 elements. Useful for output functions, but is also used for the insignificant
        # improvement stopping criterion.
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
        
        # call output function, but neglect the returned stop flag
        if self.OutputFcn:
            self.OutputFcn(self)
    
    
    def __evaluateDE(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        n = self.D
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurennetFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.z[i,:]) for i in range(n) )
        else:
            self.CurennetFitness[:] = [self.ObjectiveFcn(self.z[i,:]) for i in range(n)]
    
    
    def mutation(self):
        stop = 0
        print('mutation')
        stop = stop + 1
        print(stop)      
        if(stop>30):
            return False
        
        #return self.V
    
    def wtf(self):
        print('wtf')

    
    def optimize( self ):
        
        print("test")
        """ Runs the iterative process on the initialized swarm. """
        nVars = self.nVars
        
        #find radius
        k = round(self.D * self.Nf) # Neighborhood size
        print("To k einai: %f" %k)
        # start the iteration
        doStop = False
        
        while not doStop:
            self.Iteration += 1
            #Calculate weight w through eq. (6) => w = w min + (w max − w min ) *(t − 1) / (I max − 1)
            
            w = self.w_min+(self.w_max-self.w_min)*(self.Iteration-1)/(self.MaxIterations-1) 
          
             #-------------------start of loop through ensemble------------------------
            for i in range(0, self.D):
                
                #Mutate using eq. (3)–(5)!new vectoryi
              
                # result_array = np.array([l for l in range((i-k),(i+k))])  
                
                # result_array[result_array<0] = result_array[result_array<0] + self.D
                
                # result_array[result_array>(self.D-1)]=result_array[result_array>(self.D-1)] -self.D 
                
                # print(result_array)
                
                self.mutation()
                
                
                
                # # find neighbors
                # neighbors = np.random.choice( self.D-1, size=self.AdaptiveNeighborhoodSize, replace=False)
                # neighbors[neighbors >= i] += 1; # do not select itself, i.e., index i
                
                # bInd = self.PreviousBestFitness[neighbors].argmin()
                # bestNeighbor = neighbors[bInd]
                
                
                # # update velocity
                # randSelf = np.random.rand(nVars)
                # randSocial = np.random.rand(nVars)
                # self.Velocity[i,:] = self.AdaptiveInertia * self.Velocity[i,:] \
                #     + selfWeight * randSelf * (self.PreviousBestPosition[i,:] - self.Swarm[i,:]) \
                #     + socialWeight * randSocial * (self.PreviousBestPosition[bestNeighbor,:] - self.Swarm[i,:])
                
                # # update position
                # self.Swarm[i,:] += self.Velocity[p,:]
                
            #     # check bounds violation
            #     posInvalid = self.Swarm[i,:] < self.LowerBounds
            #     self.Swarm[i,posInvalid] = self.LowerBounds[posInvalid]
            #     self.Velocity[i,posInvalid] = 0.0
                
            #     posInvalid = self.Swarm[i,:] > self.UpperBounds
            #     self.Swarm[i,posInvalid] = self.UpperBounds[posInvalid]
            #     self.Velocity[i,posInvalid] = 0.0
            
            
            # # calculate new fitness & update best
            # self.__evaluateDE()
            # particlesProgressed = self.CurennetFitness < self.PreviousBestFitness
            # self.PreviousBestPosition[particlesProgressed, :] = self.Swarm[particlesProgressed, :]
            # self.PreviousBestFitness[particlesProgressed] = self.CurennetFitness[particlesProgressed]
            
            # #calculate fitness
            # self.__evaluateGen()
            # # find chromosomes tha has been improved and replace the old values with the new
            # genProgressed = self.CurrentGenFitness < self.PreviousBestFitness
            # self.PreviousBestPosition[genProgressed, :] = self.u[genProgressed, :]
            # self.z[genProgressed, :] = self.u[genProgressed, :]
            # self.PreviousBestFitness[genProgressed] = self.CurrentGenFitness[genProgressed]
            
            # # update global best, adaptive neighborhood size and stall counter
            # newBestInd = self.CurrentGenFitness.argmin()
            # newBestFit = self.CurrentGenFitness[newBestInd]
            
            # if newBestFit < self.GlobalBestFitness:
            #     self.GlobalBestFitness = newBestFit
            #     self.GlobalBestPosition = self.z[newBestInd, :].copy()
                
            #     self.StallCounter = max(0, self.StallCounter-1)
            #     # calculate remaining only once when fitness is improved to save some time
            #     # useful for the plots created
            #     [self.newOrder,self.remaining] = extract_obj(self)
            # else:
            #     self.StallCounter += 1
                
            # first element of self.GlobalBestSoFarFitnesses is for self.Iteration == -1
            self.GlobalBestSoFarFitnesses[self.Iteration+1] = self.GlobalBestFitness
            
            # run output function and stop if necessary
            if self.OutputFcn and self.OutputFcn(self):
                self.StopReason = 'OutputFcn requested to stop.'
                doStop = True
                continue
            
            # stop if max iterations
            if self.Iteration >= self.MaxIterations-1:
                self.StopReason = 'MaxIterations reached.'
                doStop = True
                continue
            
            # stop if insignificant improvement
            if self.Iteration > self.MaxStallIterations:
                # The minimum global best fitness is the one stored in self.GlobalBestSoFarFitnesses[self.Iteration+1]
                # (only updated if newBestFit is less than the previously stored). The maximum (may be equal to the 
                # current) is the one  in self.GlobalBestSoFarFitnesses MaxStallIterations before.
                minBestFitness = self.GlobalBestSoFarFitnesses[self.Iteration+1]
                maxPastBestFit = self.GlobalBestSoFarFitnesses[self.Iteration+1-self.MaxStallIterations]
                if (maxPastBestFit == 0.0) and (minBestFitness < maxPastBestFit):
                    windowProgress = np.inf  # don't stop
                elif (maxPastBestFit == 0.0) and (minBestFitness == 0.0):
                    windowProgress = 0.0  # not progressed
                else:
                    windowProgress = abs(minBestFitness - maxPastBestFit) / abs(maxPastBestFit)
                if windowProgress <= self.FunctionTolerance:
                    self.StopReason = 'Population did not improve significantly the last MaxStallIterations.'
                    doStop = True
            
        
        # print stop message
        print('Algorithm stopped after {} iterations. Best fitness attained: {}'.format(
                self.Iteration+1,self.GlobalBestFitness))
        print(f'Stop reason: {self.StopReason}')
        
            
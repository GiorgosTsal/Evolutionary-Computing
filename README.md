# Evolutionary-Computing

### Use of evolutionary algorithms to optimize wood stock management.

Even a timber factory that receives orders to supply a certain number of pieces of wood. The goal is to make the best use of the factory's current stock of wood so that the stock remaining after removing the pieces of order can be reused on a subsequent order without losing a large area of ​​wood. This practically means that pieces of wood left after cutting should be as solid as possible.

## Use of evolutionary algorithms

* [Particle Swarm Optimization, PSO with dynamic neighborhood topology](https://en.wikipedia.org/wiki/Particle_swarm_optimization) 
* [Differential Evolution, DE with global and local neighborhood topologies](https://en.wikipedia.org/wiki/Differential_evolution) 

## Use of optimization algorithms

* [Simplex](https://en.wikipedia.org/wiki/Simplex_algorithm) 
* [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) 
* [SLSQP](http://degenerateconic.com/slsqp/) 
* [Pattern search](https://en.wikipedia.org/wiki/Pattern_search_(optimization)) 

### Prerequisites

* matplotlib
* numpy 
* https://github.com/Toblerity/Shapely. (The package is based on the GEOS C ++ library, which must be installed)
* Noisyopt: A python library for optimizing noisy functions (https://github.com/andim/noisyopt)

```
pip install Shapely
```
or
```
conda install -c ”conda-forge” shapely
```

* https://pypi.org/project/descartes/ . (Use geometric objects as matplotlib paths and patches)

```
pip install descartes
```
or
```
conda install -c conda-forge descartes 
```

Noisyopt is on PyPI so you can install it using:
``` pip install noisyopt ```

## Running the tests

Each folder has a readme on how to run the programs 

## Authors

* **Giorgos Tsalidis** - [LinkedIn ](https://gr.linkedin.com/in/tsalidis-giorgos)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

[1] S. K. Mylonas, D. G. Stavrakoudis, J. B. Theocharis, and P. A. Mastorocostas, “A Region-Based
GeneSIS Segmentation Algorithm for the Classification of Remotely Sensed Images,” Remote
Sensing, vol. 7, no. 3, pp. 2474–2508, Mar. 2015. Online link: http://www.mdpi.com/2072-4292/
7/3/2474 (page 1)

[2] S. Das, A. Abraham, U. Chakraborty, and A. Konar, “Differential Evolution Using a Neighborhood-Based Mutation Operator,” IEEE Transactions on Evolutionary Computation, vol. 13, no. 3,
pp. 526–553, Jun. 2009. (page 3)

[3] S. Das and S. Sil, “Kernel-induced fuzzy clustering of image pixels with an improved differential
evolution algorithm,” Information Sciences, vol. 180, no. 8, pp. 1237–1256, Apr. 2010. Online link:
http://www.sciencedirect.com/science/article/pii/S0020025509005192 (p. 3)

[4] M. Dorigo and T. Stützle, Ant Colony Optimization, ser. Bradford Books. Cambridge, MA, USA:
MIT Press, Jun. 2004. (page 6)

### Libraries and tools:

* https://github.com/Toblerity/Shapely
* https://github.com/andim/noisyopt

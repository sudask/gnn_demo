# gnn_demo
a simple gnn to solve a bivariate function fitting problem
![alt text](image.png)

12.2 
Now using normalized distance between target point and grid points as node feature (of course function value on grids included). Loss will converge in 20 epochs. Using shuffle to avoid convergece to local minimum. 
Not very accurate on points between grid points.
Move plot from utils/ to root directory, so that it is much easier to use.

12.3
Abort strategy of using top K feature. Because when a grid point has a very small func val, its feature will be small whether it is near target point or not.
In config, give an appropriate learning rate under different GRID_SIZE.
Now it is a big problem how to combine two grids of different scale and different observing operator.

Now a potential approach is: building two models, using weighted sum of these two models. To some extent, we can claim that model using grid bigger than the other will have a better result, and the mixture of a better result and a poorer one seems to generate a 'middle' result, which is worse than the better one. But the key point is there are two different observing operator, which leads to the probablility that a model gives a better result with a smaller grid in some points.

12.5
Now the CompleteGNN to solve the full question is set. Just run main.py, you will get a loss curve and a figure. The potential approach proves good. Although the two grids generate two results in which one is better than the other, mixture of these two gives a lower loss averaged on all training data. 0.001 is an appropriate learning rate for now. But I dont know whether the result itself is good -- will it outperform bilinear interpolation?

12.18
Changed the smaller grid to ensure that these two grids do not interlap. Modified the approach to use different grids. Intuitively, it gives a better result when use two grids to simulate. But in previous version, it was not the case. There was also no testing case with one grid and use observations, so when results appeared to be poor using two grids, we cannot figure out what's the problem from.
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
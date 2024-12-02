# gnn_demo
a simple gnn to solve a bivariate function fitting problem
![alt text](image.png)

12.2 
Now using normalized distance between target point and grid points as node feature (of course function value on grids included). Loss will converge in 20 epochs. Using shuffle to avoid convergece to local minimum. 
Not very accurate on points between grid points.
Move plot from utils/ to root directory, so that it is much easier to use.
// Gmsh project created on Fri Feb 23 10:51:28 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {-1.5, -0.3, 0, 1};
//+
Point(2) = {-1.5, 0.3, 0, 1};
//+
Point(3) = {-1.5, 0.5, 2.5, 1};
//+
Point(4) = {-1.5, -0.3, 2.5, 1};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Curve Loop(2) = {3, 4, 1, 2};
//+
Plane Surface(1) = {2};
//+
Extrude {3, 0, 0} {
  Surface{1}; 
}

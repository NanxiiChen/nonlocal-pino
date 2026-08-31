// Rounded finite-width notch opening at the left boundary.
// The rightmost point of the semicircular tip is (0.5, 0.5).

Lx = DefineNumber[1.0, Name "Lx"];
Ly = DefineNumber[1.0, Name "Ly"];
tip_x = DefineNumber[0.5, Name "tip_x"];
center_y = DefineNumber[0.5, Name "center_y"];
r_tip = DefineNumber[0.024, Name "r_tip"];
lc_far = DefineNumber[0.02, Name "lc_far"];
lc_near = DefineNumber[0.004, Name "lc_near"];

center_x = tip_x - r_tip;

Point(1) = {0, 0, 0, lc_far};
Point(2) = {Lx, 0, 0, lc_far};
Point(3) = {Lx, Ly, 0, lc_far};
Point(4) = {0, Ly, 0, lc_far};
Point(5) = {0, center_y + r_tip, 0, lc_near};
Point(6) = {center_x, center_y + r_tip, 0, lc_near};
Point(7) = {tip_x, center_y, 0, lc_near};
Point(8) = {center_x, center_y - r_tip, 0, lc_near};
Point(9) = {0, center_y - r_tip, 0, lc_near};
Point(10) = {center_x, center_y, 0, lc_near};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Circle(6) = {6, 10, 7};
Circle(7) = {7, 10, 8};
Line(8) = {8, 9};
Line(9) = {9, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
Plane Surface(1) = {1};

Physical Curve("bottom", 1) = {1};
Physical Curve("right", 2) = {2};
Physical Curve("top", 3) = {3};
Physical Curve("left_upper", 4) = {4};
Physical Curve("notch_upper", 5) = {5};
Physical Curve("notch_tip", 6) = {6, 7};
Physical Curve("notch_lower", 7) = {8};
Physical Curve("left_lower", 8) = {9};
Physical Point("anchor", 9) = {1};
Physical Surface("domain", 10) = {1};

Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = lc_near;
Mesh.CharacteristicLengthMax = lc_far;

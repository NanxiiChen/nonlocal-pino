// 2D rectangular plate with circular hole — x longer than y
// Domain: [0, Lx] x [0, Ly] with hole radius a at (Lx/2, Ly/2)

Lx = 1.5;    // plate length in x (longer)
Ly = 1.0;    // plate length in y
a  = 0.2;    // hole radius
lc0 = 0.04;  // mesh size far from hole
lc1 = 0.015; // mesh size near hole (stress concentration)
cx = Lx/2;   // hole centre x
cy = Ly/2;   // hole centre y

// ── Outer rectangle (CCW) ─────────────────────────────────────────
Point(1) = {0,  0,  0, lc0};
Point(2) = {Lx, 0,  0, lc0};
Point(3) = {Lx, Ly, 0, lc0};
Point(4) = {0,  Ly, 0, lc0};

Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // right  (loaded edge — u_x = u_load)
Line(3) = {3, 4};  // top
Line(4) = {4, 1};  // left   (clamped edge — u_x = u_y = 0)

// ── Circular hole (CW for subtraction) ────────────────────────────
Point(5)  = {cx + a, cy,      0, lc1};
Point(6)  = {cx,      cy + a, 0, lc1};
Point(7)  = {cx - a, cy,      0, lc1};
Point(8)  = {cx,      cy - a, 0, lc1};
Point(9)  = {cx,      cy,     0, lc1};

Circle(5) = {5, 9, 6};
Circle(6) = {6, 9, 7};
Circle(7) = {7, 9, 8};
Circle(8) = {8, 9, 5};

// ── Surface with hole ─────────────────────────────────────────────
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

// ── Physical groups ───────────────────────────────────────────────
Physical Curve("left", 10)   = {4};
Physical Curve("right", 20)  = {2};
Physical Curve("top", 30)    = {3};
Physical Curve("bottom", 40) = {1};
Physical Curve("hole", 50)   = {5, 6, 7, 8};
Physical Surface("domain", 100) = {1};

// Mesh size field: refine near hole
Field[1] = Distance;
Field[1].CurvesList = {5, 6, 7, 8};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc1;
Field[2].LcMax = lc0;
Field[2].DistMin = 0.02;
Field[2].DistMax = 0.2;

Background Field = 2;

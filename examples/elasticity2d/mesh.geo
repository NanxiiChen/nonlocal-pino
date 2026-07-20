// 2D rectangular plate with elliptical hole — steel, non-dimensional
// Physical: 2m × 1.5m plate, elliptical hole 0.1m × 0.3m
// Non-dimensional: Lx=1.0, Ly=0.75, a_x=0.05, a_y=0.15

Lx = 1.0;    // plate length in x (non-dim)
Ly = 0.75;   // plate length in y
a_x = 0.10;  // semi-axis in x (parallel to loading)
a_y = 0.15;  // semi-axis in y (perpendicular to loading)
lc0 = 0.015; // mesh size far from hole
lc1 = 0.004; // mesh size near hole (stress concentration)
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

// ── Elliptical hole (CW for subtraction) ─────────────────────────
// Major axis = vertical (a_y > a_x) → stress concentrates at top/bottom
Point(5)  = {cx + a_x, cy,       0, lc1};  // right
Point(6)  = {cx,       cy + a_y, 0, lc1};  // top (major-axis point)
Point(7)  = {cx - a_x, cy,       0, lc1};  // left
Point(8)  = {cx,       cy - a_y, 0, lc1};  // bottom
Point(9)  = {cx,       cy,       0, lc1};  // centre

Ellipse(5) = {5, 9, 6, 6};  // NE: right → top
Ellipse(6) = {6, 9, 6, 7};  // NW: top → left
Ellipse(7) = {7, 9, 6, 8};  // SW: left → bottom
Ellipse(8) = {8, 9, 6, 5};  // SE: bottom → right

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

with Ada.Numerics;

package Geometric_Primitives is
   pragma Pure;

   subtype Coordinate is Long_Float range -1.0e12 .. 1.0e12;
   
   type Point is record
      X, Y : Coordinate;
   end record;

   type Line is record
      A, B, C : Coordinate;
   end record
     with Dynamic_Predicate => (Line.A /= 0.0 or Line.B /= 0.0);

   type Circle is record
      Center : Point;
      Radius : Coordinate;
   end record
     with Dynamic_Predicate => Circle.Radius >= 0.0;

   type Polygon is array (Positive range <>) of Point
     with Dynamic_Predicate => Polygon'Length >= 3;

   -- Line construction from two points
   function Create_Line (P1, P2 : Point) return Line
     with Pre => P1 /= P2;

   -- Distance between two points
   function Distance (P1, P2 : Point) return Coordinate
     with Post => Distance'Result >= 0.0;

   -- Point-to-line distance
   function Distance (P : Point; L : Line) return Coordinate
     with Post => Distance'Result >= 0.0;

   -- Circle circumference
   function Circumference (C : Circle) return Coordinate
     with Post => Circumference'Result >= 0.0;

   -- Polygon perimeter
   function Perimeter (P : Polygon) return Coordinate
     with Post => Perimeter'Result >= 0.0;

   -- Collinearity check
   function Are_Collinear (P1, P2, P3 : Point) return Boolean;

   -- Line intersection
   function Intersection (L1, L2 : Line) return Point
     with Pre => abs (L1.A * L2.B - L2.A * L1.B) > 1.0e-6;

   -- Polygon area (Shoelace formula)
   function Area (P : Polygon) return Coordinate
     with Post => Area'Result >= 0.0;

private
   function Determinant (P1, P2 : Point) return Coordinate is
     (P1.X * P2.Y - P1.Y * P2.X);
end Geometric_Primitives;

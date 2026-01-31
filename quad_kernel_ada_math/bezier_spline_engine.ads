package Bezier_Spline_Engine is

   type Real is digits 15 range -1.0e15 .. 1.0e15;
   subtype Unit_Interval is Real range 0.0 .. 1.0;

   type Point is record
      X, Y : Real;
   end record;

   type Bezier_Segment is array (0 .. 3) of Point;
   type Bezier_Spline is array (Natural range <>) of Bezier_Segment;

   function Evaluate_Segment (S : Bezier_Segment; T : Unit_Interval) return Point
     with Post =>
           (if T = 0.0 then Evaluate_Segment'Result = S(0)) and
           (if T = 1.0 then Evaluate_Segment'Result = S(3));

   function Evaluate_Spline (Spline : Bezier_Spline; T : Real) return Point
     with Pre => Spline'Length >= 1 and T >= 0.0 and T <= Real(Spline'Length);

end Bezier_Spline_Engine;

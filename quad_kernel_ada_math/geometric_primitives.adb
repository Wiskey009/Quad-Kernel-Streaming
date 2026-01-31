with Ada.Numerics.Generic_Elementary_Functions;

package body Geometric_Primitives is
   
   package Coord_Math is new Ada.Numerics.Generic_Elementary_Functions(Coordinate);
   use Coord_Math;

   function Create_Line (P1, P2 : Point) return Line is
      A : constant Coordinate := P2.Y - P1.Y;
      B : constant Coordinate := P1.X - P2.X;
      C : constant Coordinate := Determinant(P2, P1);
   begin
      return (A, B, C);
   end Create_Line;

   function Distance (P1, P2 : Point) return Coordinate is
      DX : constant Coordinate := P2.X - P1.X;
      DY : constant Coordinate := P2.Y - P1.Y;
   begin
      return Sqrt(DX * DX + DY * DY);
   end Distance;

   function Distance (P : Point; L : Line) return Coordinate is
      Num : constant Coordinate := abs(L.A * P.X + L.B * P.Y + L.C);
      Den : constant Coordinate := Sqrt(L.A * L.A + L.B * L.B);
   begin
      return Num / Den;
   end Distance;

   function Circumference (C : Circle) return Coordinate is
   begin
      return 2.0 * Ada.Numerics.Pi * C.Radius;
   end Circumference;

   function Perimeter (P : Polygon) return Coordinate is
      Total : Coordinate := 0.0;
   begin
      for I in P'First .. P'Last - 1 loop
         Total := Total + Distance(P(I), P(I + 1));
      end loop;
      Total := Total + Distance(P(P'Last), P(P'First));
      return Total;
   end Perimeter;

   function Are_Collinear (P1, P2, P3 : Point) return Boolean is
      V1 : constant Point := (P2.X - P1.X, P2.Y - P1.Y);
      V2 : constant Point := (P3.X - P1.X, P3.Y - P1.Y);
   begin
      return abs(Determinant(V1, V2)) < 1.0e-6;
   end Are_Collinear;

   function Intersection (L1, L2 : Line) return Point is
      Det : constant Coordinate := L1.A * L2.B - L2.A * L1.B;
      X   : Coordinate;
      Y   : Coordinate;
   begin
      X := (L2.C * L1.B - L1.C * L2.B) / Det;
      Y := (L1.C * L2.A - L2.C * L1.A) / Det;
      return (X, Y);
   end Intersection;

   function Area (P : Polygon) return Coordinate is
      Total : Coordinate := 0.0;
      Prev  : Point := P(P'Last);
   begin
      for I in P'Range loop
         declare
            Current : constant Point := P(I);
         begin
            Total := Total + Determinant(Prev, Current);
            Prev := Current;
         end;
      end loop;
      return abs(Total) / 2.0;
   end Area;
   
end Geometric_Primitives;

with Geometric_Primitives; use Geometric_Primitives;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics;

procedure Test_Geometric_Primitives is
   P1 : constant Point := (0.0, 0.0);
   P2 : constant Point := (1.0, 0.0);
   P3 : constant Point := (0.0, 1.0);
   L1 : constant Geometric_Primitives.Line := Create_Line(P1, P2);
   C  : constant Circle := ((0.0, 0.0), 1.0);
begin
   if Distance(P1, P2) /= 1.0 then
      Put_Line("Distance failed");
   else
      Put_Line("Distance passed");
   end if;

   if not Are_Collinear(P1, P2, (2.0, 0.0)) then
      Put_Line("Collinearity failed");
   else
      Put_Line("Collinearity passed");
   end if;

   if abs(Area([P1, P2, P3]) - 0.5) > 1.0e-6 then
      Put_Line("Area failed: " & Area([P1, P2, P3])'Img);
   else
      Put_Line("Area passed");
   end if;

   if abs(Circumference(C) - 2.0 * Ada.Numerics.Pi) > 1.0e-6 then
      Put_Line("Circumference failed");
   else
      Put_Line("Circumference passed");
   end if;

   Put_Line("Geometric Primitives tests finished.");
end Test_Geometric_Primitives;

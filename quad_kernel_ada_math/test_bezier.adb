with Ada.Text_IO; use Ada.Text_IO;
with Bezier_Spline_Engine; use Bezier_Spline_Engine;

procedure Test_Bezier is
   --  Test 1: Linear Segment (P0 to P3)
   Linear : Bezier_Segment := ((0.0, 0.0), (1.0/3.0, 1.0/3.0), (2.0/3.0, 2.0/3.0), (1.0, 1.0));
   --  Test 2: Degenerate (All points same)
   Point_Test : Bezier_Segment := ((5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0));
   Spline : Bezier_Spline(0..1) := (Linear, Point_Test);
begin
   if Evaluate_Segment(Linear, 0.0) /= (0.0, 0.0) then
      Put_Line("Start Failed");
   else
      Put_Line("Start Passed");
   end if;

   if Evaluate_Segment(Linear, 1.0) /= (1.0, 1.0) then
      Put_Line("End Failed");
   else
      Put_Line("End Passed");
   end if;

   declare
     P : constant Point := Evaluate_Spline(Spline, 2.0);
   begin
      if P /= (5.0, 5.0) then
         Put_Line("Spline End Failed: " & P.X'Img & P.Y'Img);
      else
         Put_Line("Spline End Passed");
      end if;
   end;

   Put_Line("Bezier tests finished.");
end Test_Bezier;

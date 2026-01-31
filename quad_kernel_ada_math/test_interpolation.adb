with Ada.Text_IO; use Ada.Text_IO;
with Interpolation_Methods;

procedure Test_Interpolation is
   package Real_Interp is new Interpolation_Methods (Long_Float);
   use Real_Interp;

   Points : Data_Points(5) := (
     Length => 5,
     X => (1.0, 2.0, 3.0, 4.0, 5.0),
     Y => (1.0, 8.0, 27.0, 64.0, 125.0));

   Coeffs : Spline_Coeffs(1..4);
begin
   -- Test linear interpolation
   if Linear_Interpolate(Points, 1.5) /= 4.5 then
      Put_Line("Linear 1.5 failed");
   else
      Put_Line("Linear 1.5 passed");
   end if;

   -- Cubic spline tests
   Compute_Spline_Coefficients(Points, Coeffs);
   declare
      Val : constant Long_Float := Evaluate_Spline(Points, Coeffs, 1.0);
   begin
      if abs(Val - 1.0) > 1.0e-6 then
         Put_Line("Spline 1.0 failed: " & Val'Img);
      else
         Put_Line("Spline 1.0 passed");
      end if;
   end;

   Put_Line("All interpolation tests finished.");
end Test_Interpolation;

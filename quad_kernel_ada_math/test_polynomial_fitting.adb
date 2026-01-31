with Ada.Text_IO; use Ada.Text_IO;
with Polynomial_Fitting_Solver;

procedure Test_Polynomial_Fitting is
   package Float_Solver is new Polynomial_Fitting_Solver(Float);
   use Float_Solver;
   use Float_Solver.Real_Arrays;
   
   -- Points {(0,1), (1,3), (2,5)} -> y = 2x + 1
   Points : constant Point_List(3) := (
      Length => 3,
      X => (0.0, 1.0, 2.0),
      Y => (1.0, 3.0, 5.0)
   );
begin
   declare
      Coeffs : constant Real_Vector := Least_Squares_Fit(Points, 1);
   begin
      Put_Line("Linear fit:");
      -- Expect [1.0, 2.0]
      for I in Coeffs'Range loop
         Put_Line(" C(" & I'Img & ") =" & Coeffs(I)'Img);
      end loop;
      
      if abs(Coeffs(0) - 1.0) > 1.0e-5 or abs(Coeffs(1) - 2.0) > 1.0e-5 then
         Put_Line("Linear fit failed");
      else
         Put_Line("Linear fit passed");
      end if;
   end;

   Put_Line("Polynomial fitting tests finished.");
end Test_Polynomial_Fitting;

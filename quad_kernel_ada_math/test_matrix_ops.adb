with Ada.Text_IO; use Ada.Text_IO;
with Matrix_Operations_Kernel;

procedure Test_Matrix_Ops is
   package Mat_Op is new Matrix_Operations_Kernel(Float, 2);
   use Mat_Op;
   
   A : Matrix := ((1.0, 2.0), (3.0, 4.0));
   B : Matrix := ((5.0, 6.0), (7.0, 8.0));
   C : Matrix;
begin
   -- Test Addition
   C := A + B;
   if C(1,1) /= 6.0 or C(2,2) /= 12.0 then
      Put_Line("Addition failed");
   else
      Put_Line("Addition passed");
   end if;

   -- Test Multiplication
   -- [1 2] [5 6]   [1*5+2*7 1*6+2*8]   [19 22]
   -- [3 4] [7 8] = [3*5+4*7 3*6+4*8] = [43 50]
   C := A * B;
   if C(1,1) /= 19.0 or C(2,2) /= 50.0 then
      Put_Line("Multiplication failed: " & C(1,1)'Img & C(2,2)'Img);
   else
      Put_Line("Multiplication passed");
   end if;

   Put_Line("Matrix operations tests finished.");
end Test_Matrix_Ops;

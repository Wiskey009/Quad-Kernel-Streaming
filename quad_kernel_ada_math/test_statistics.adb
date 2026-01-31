with Ada.Text_IO; use Ada.Text_IO;
with Statistical_Analysis_Module; use Statistical_Analysis_Module;

procedure Test_Statistics is
   type Float_Array is array(Positive range <>) of Float;
   function Float_Mean is new Statistical_Analysis_Module.Mean(Float, Float_Array);
   function Float_Var is new Statistical_Analysis_Module.Variance(Float, Float_Array);
   
   Data : constant Float_Array(1..4) := [1.0, 2.0, 3.0, 4.0];
begin
   declare
      Mu : constant Float := Float_Mean(Data);
   begin
      if abs(Mu - 2.5) > 1.0e-6 then
         Put_Line("Mean failed: " & Mu'Img);
      else
         Put_Line("Mean passed");
      end if;
   end;

   declare
      Var : constant Float := Float_Var(Data);
   begin
      if abs(Var - 1.666666) > 1.0e-5 then
         Put_Line("Variance failed: " & Var'Img);
      else
         Put_Line("Variance passed");
      end if;
   end;

   declare
      Uniform : constant Probability_Array := [0.25, 0.25, 0.25, 0.25];
      H : constant Real := Entropy(Uniform);
   begin
      -- log(4) approx 1.386294
      if abs(H - 1.38629436) > 1.0e-6 then
         Put_Line("Entropy failed: " & H'Img);
      else
         Put_Line("Entropy passed");
      end if;
   end;

   Put_Line("Statistical tests finished.");
end Test_Statistics;

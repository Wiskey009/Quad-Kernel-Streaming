with Ada.Text_IO; use Ada.Text_IO;
with Convolution_Correlation;

procedure Test_Convolution is
   type Float_Array is array (Integer range <>) of Float;
   type Float_Matrix is array (Integer range <>, Integer range <>) of Float;

   package Real_Conv is new Convolution_Correlation.Generic_Convolution
     (Real => Float,
      Index_Type => Integer,
      Signal_1D => Float_Array,
      Signal_2D => Float_Matrix);
   use Real_Conv;

   -- Delta kernel test: convolution with [0, 1, 0] should return the original signal shifted/centered
   Impulse : constant Float_Array := (0.0, 1.0, 0.0);
   Signal : constant Float_Array := (1.0, 2.0, 3.0, 4.0);
begin
   declare
      Res : constant Float_Array := Convolve_1D(Signal, Impulse);
   begin
      -- Length = 4 + 3 - 1 = 6
      if Res'Length /= 6 then
         Put_Line("1D length failure: " & Res'Length'Img);
      else
         Put_Line("1D length passed");
      end if;
      
      -- Res should be approx [0, 1, 2, 3, 4, 0] depending on alignment
      Put_Line("1D Res(2): " & Res(Signal'First + 1)'Img); -- Should be 1.0 if shifted
   end;

   declare
      Sobel : constant Float_Matrix := ((1.0, 0.0, -1.0), (2.0, 0.0, -2.0), (1.0, 0.0, -1.0));
      Image : constant Float_Matrix := ((1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 1.0));
      Edges : constant Float_Matrix := Convolve_2D(Image, Sobel);
   begin
      Put_Line("2D Central Edges(2,2): " & Edges(2,2)'Img);
   end;

   Put_Line("Convolution tests finished.");
end Test_Convolution;

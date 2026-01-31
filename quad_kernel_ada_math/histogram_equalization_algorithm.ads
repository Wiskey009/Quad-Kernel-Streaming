package Histogram_Equalization_Algorithm is
   subtype Pixel_Value is Integer range 0..255;
   type Image_Matrix is array (Natural range <>, Natural range <>) of Pixel_Value;
   type Histogram is array (Pixel_Value) of Natural;

   --  Main equalization procedure
   procedure Adaptive_Equalize 
     (Input : in Image_Matrix;
      Output : out Image_Matrix;
      Window_Radius : in Positive)
   with
     Pre => (Input'Length(1) > 0 and Input'Length(2) > 0)
       and then (Input'First(1) = Output'First(1))
       and then (Input'Last(1) = Output'Last(1))
       and then (Input'First(2) = Output'First(2))
       and then (Input'Last(2) = Output'Last(2));
        
end Histogram_Equalization_Algorithm;

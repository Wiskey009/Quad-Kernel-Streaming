package body Histogram_Equalization_Algorithm is

   function Compute_CDF (H : Histogram) return Histogram is
      CDF : Histogram := (others => 0);
   begin
      CDF(0) := H(0);
      for K in 1..255 loop
         CDF(K) := CDF(K-1) + H(K);
      end loop;
      return CDF;
   end Compute_CDF;

   function Remap_Value (CDF : Histogram; Value : Pixel_Value) return Pixel_Value is
      Scaled : Natural;
   begin
      if CDF(255) = 0 then
         return 0;
      end if;
      Scaled := (255 * CDF(Value)) / CDF(255);
      return Pixel_Value(Scaled);
   end Remap_Value;

   procedure Adaptive_Equalize 
     (Input : in Image_Matrix;
      Output : out Image_Matrix;
      Window_Radius : in Positive) 
   is
   begin
      for I in Input'Range(1) loop
         for J in Input'Range(2) loop
            declare
               H : Histogram := (others => 0);
               --  Determine bounds of the local window
               Start_Row : constant Integer := Integer'Max(Input'First(1), Integer(I) - Window_Radius);
               End_Row : constant Integer := Integer'Min(Input'Last(1), Integer(I) + Window_Radius);
               Start_Col : constant Integer := Integer'Max(Input'First(2), Integer(J) - Window_Radius);
               End_Col : constant Integer := Integer'Min(Input'Last(2), Integer(J) + Window_Radius);
            begin
               --  Build local histogram
               for X in Start_Row..End_Row loop
                  for Y in Start_Col..End_Col loop
                     H(Input(X,Y)) := H(Input(X,Y)) + 1;
                  end loop;
               end loop;

               --  Equalization core
               declare
                  CDF : constant Histogram := Compute_CDF(H);
               begin
                  Output(I,J) := Remap_Value(CDF, Input(I,J));
               end;
            end;
         end loop;
      end loop;
   end Adaptive_Equalize;

end Histogram_Equalization_Algorithm;

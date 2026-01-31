with Ada.Text_IO; use Ada.Text_IO;
with Histogram_Equalization_Algorithm; use Histogram_Equalization_Algorithm;

procedure Test_Histogram is
   Test_Image : Image_Matrix(0..3, 0..3) := 
     ((50, 100, 150, 200),
      (50, 100, 150, 200),
      (50, 100, 150, 200),
      (50, 100, 150, 200));
   Result : Image_Matrix(0..3, 0..3);
begin
   --  Uniform regions test
   Adaptive_Equalize(Test_Image, Result, 5); -- uses global stats since window > image
   
   for I in Result'Range(1) loop
      for J in Result'Range(2) loop
         Put(Result(I,J)'Img & " ");
      end loop;
      New_Line;
   end loop;

   Put_Line("Histogram Equalization tests finished.");
end Test_Histogram;

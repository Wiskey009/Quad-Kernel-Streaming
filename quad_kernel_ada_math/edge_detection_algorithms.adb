with Ada.Numerics.Elementary_Functions;

package body Edge_Detection_Algorithms is
   use Ada.Numerics.Elementary_Functions;

   Sobel_X : constant Kernel_3x3 := ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1));
   Sobel_Y : constant Kernel_3x3 := ((-1, -2, -1), (0, 0, 0), (1, 2, 1));
   Laplacian_Kernel : constant Kernel_3x3 := ((0, -1, 0), (-1, 4, -1), (0, -1, 0));

   function Clamp (Value : Integer) return Pixel_Value is
   begin
      if Value < 0 then
         return 0;
      elsif Value > 255 then
         return 255;
      else
         return Value;
      end if;
   end Clamp;

   function Sobel_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value is
      Gx : Integer := 0;
      Gy : Integer := 0;
   begin
      for I in -1 .. 1 loop
         for J in -1 .. 1 loop
            Gx := Gx + Sobel_X(I, J) * Img(X+I, Y+J);
            Gy := Gy + Sobel_Y(I, J) * Img(X+I, Y+J);
         end loop;
      end loop;

      -- Scale down as Sobel can reach values way above 255
      -- Max Gx is 4 * 255 = 1020
      -- Max G is sqrt(1020^2 + 1020^2) approx 1442
      -- Dividing by 4 brings it back to approx 360 which still needs clamping.
      return Clamp (Integer(Sqrt (Float(Gx**2 + Gy**2)) / 4.0));
   end Sobel_Operator;

   function Laplacian_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value is
      Sum : Integer := 0;
   begin
      for I in -1 .. 1 loop
         for J in -1 .. 1 loop
            Sum := Sum + Laplacian_Kernel(I, J) * Img(X+I, Y+J);
         end loop;
      end loop;
      return Clamp (Sum);
   end Laplacian_Operator;

end Edge_Detection_Algorithms;

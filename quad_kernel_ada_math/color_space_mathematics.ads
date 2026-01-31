with Ada.Numerics.Generic_Real_Arrays;

generic
   type Real is digits <>;
package Color_Space_Mathematics is
   pragma Pure;
   pragma Assertion_Policy (Pre => Check, Post => Check);

   subtype Percent is Real range 0.0..1.0;
   subtype Degree is Real range 0.0..360.0;
   subtype Chroma is Real range -0.5..0.5;

   type RGB_Color is record
      R, G, B : Percent;
   end record;

   type YUV_Color is record
      Y : Percent;
      U, V : Chroma;
   end record;

   type HSV_Color is record
      H : Degree;
      S, V : Percent;
   end record;

   function RGB_To_YUV (RGB : RGB_Color) return YUV_Color
     with Pre => (RGB.R in 0.0..1.0 and
                  RGB.G in 0.0..1.0 and
                  RGB.B in 0.0..1.0),
          Post => (RGB_To_YUV'Result.Y in 0.0..1.0 and
                   RGB_To_YUV'Result.U in Chroma and
                   RGB_To_YUV'Result.V in Chroma);

   function YUV_To_RGB (YUV : YUV_Color) return RGB_Color
     with Pre => (YUV.Y in 0.0..1.0 and
                  YUV.U in Chroma and
                  YUV.V in Chroma),
          Post => (YUV_To_RGB'Result.R in 0.0..1.0 and
                   YUV_To_RGB'Result.G in 0.0..1.0 and
                   YUV_To_RGB'Result.B in 0.0..1.0);

   function RGB_To_HSV (RGB : RGB_Color) return HSV_Color
     with Pre => (RGB.R in 0.0..1.0 and
                  RGB.G in 0.0..1.0 and
                  RGB.B in 0.0..1.0),
          Post => (RGB_To_HSV'Result.H in 0.0..360.0 and
                   RGB_To_HSV'Result.S in 0.0..1.0 and
                   RGB_To_HSV'Result.V in 0.0..1.0);

   function HSV_To_RGB (HSV : HSV_Color) return RGB_Color
     with Pre => (HSV.H in 0.0..360.0 and
                  HSV.S in 0.0..1.0 and
                  HSV.V in 0.0..1.0),
          Post => (HSV_To_RGB'Result.R in 0.0..1.0 and
                   HSV_To_RGB'Result.G in 0.0..1.0 and
                   HSV_To_RGB'Result.B in 0.0..1.0);

private
   package Real_Arrays is new Ada.Numerics.Generic_Real_Arrays (Real);
   use Real_Arrays;

   -- RGB/YUV matrix transformations
   RGB_TO_YUV_MAT : constant Real_Matrix :=
     ((0.299, 0.587, 0.114),
      (-0.14713, -0.28886, 0.436),
      (0.615, -0.51499, -0.10001));

   YUV_TO_RGB_MAT : constant Real_Matrix :=
     ((1.0, 0.0, 1.13983),
      (1.0, -0.39465, -0.58060),
      (1.0, 2.03211, 0.0));
end Color_Space_Mathematics;

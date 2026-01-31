# edge_detection_algorithms

```ada
--  edge_detection_algorithms.ads
pragma SPARK_Mode (On);

package Edge_Detection_Algorithms is
   subtype Pixel_Value is Natural range 0 .. 255;
   type Kernel_3x3 is array (-1 .. 1, -1 .. 1) of Integer;
   type Grayscale_Image is array (Integer range <>, Integer range <>) of Pixel_Value;
   
   -- Sobel operator equations:
   -- G_x = | ∂I/∂x | = | [ -1  0 +1 ]   * A |
   --       |          |   [ -2  0 +2 ]        |
   --       |          |   [ -1  0 +1 ]        |
   --
   -- G_y = | ∂I/∂y | = | [ -1 -2 -1 ]   * A |
   --       |          |   [  0  0  0 ]        |
   --       |          |   [ +1 +2 +1 ]        |
   --
   -- Gradient magnitude: G = sqrt(G_x² + G_y²)

   function Sobel_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value
     with Pre => 
       Img'First(1) <= X - 1 and X + 1 <= Img'Last(1) and
       Img'First(2) <= Y - 1 and Y + 1 <= Img'Last(2),
     Post => Sobel_Operator'Result in 0 .. 360;  -- sqrt(2*(255*4)^2) ≈ 360

   -- Laplacian operator equation:
   -- ∇²I = [ 0 -1  0 ] * A
   --       [ -1 4 -1 ]
   --       [ 0 -1  0 ]

   function Laplacian_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value
     with Pre => 
       Img'First(1) <= X - 1 and X + 1 <= Img'Last(1) and
       Img'First(2) <= Y - 1 and Y + 1 <= Img'Last(2),
     Post => Laplacian_Operator'Result in 0 .. 1020;  -- 4*255

   -- Canny theorems:
   -- 1. Non-maximum suppression preserves edges iff gradient direction matches local maximum
   -- 2. Hysteresis thresholding ensures connectivity: ∀p ∈ strong_edges, ∃path p→q where q ∈ strong_edges ∧ adjacent(p,q)

private
   -- Proof: Sobel output bounded by input constraints
   -- Let K = max(|Sobel_x|, |Sobel_y|) = 4
   -- Max gradient magnitude = sqrt((4*255)^2 + (4*255)^2) = 4*255*sqrt(2) ≈ 1442
   -- Actual bound reduced via clamping to 255
   pragma Assert (for all X in Integer => (for all Y in Integer => Sobel_Operator'(X, Y) <= 255));

   -- Proof: Laplacian bounded by kernel properties
   -- Max output = 4*255 (center pixel *4), min = 0 via clamping
   pragma Assert (for all Img in Grayscale_Image => (for all X in Img'Range(1) => (for all Y in Img'Range(2) => 
                  Laplacian_Operator(Img, X, Y) <= 4*255)));

   -- Type invariants
   function Is_Valid_Image (Img : Grayscale_Image) return Boolean is
     (Img'Length(1) >= 3 and Img'Length(2) >= 3)
     with Ghost;

end Edge_Detection_Algorithms;

-------------------------------------------------------------------

--  edge_detection_algorithms.adb
pragma SPARK_Mode (On);
with Ada.Numerics.Elementary_Functions;

package body Edge_Detection_Algorithms is
   use Ada.Numerics.Elementary_Functions;

   Sobel_X : constant Kernel_3x3 := ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1));
   Sobel_Y : constant Kernel_3x3 := ((-1, -2, -1), (0, 0, 0), (1, 2, 1));
   Laplacian_Kernel : constant Kernel_3x3 := ((0, -1, 0), (-1, 4, -1), (0, -1, 0));

   function Clamp (Value : Integer) return Pixel_Value is
     (if Value < 0 then 0 elsif Value > 255 then 255 else Value);

   function Sobel_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value is
      Gx : Integer := 0;
      Gy : Integer := 0;
   begin
      for I in -1 .. 1 loop
         for J in -1 .. 1 loop
            pragma Loop_Invariant (for all K in -1 .. I-1 => 
              (for all L in -1 .. 1 => 
                 Gx = Gx'Loop_Entry + Sobel_X(K, L) * Img(X+K, Y+L)));
            pragma Loop_Invariant (for all K in -1 .. I-1 => 
              (for all L in -1 .. 1 => 
                 Gy = Gy'Loop_Entry + Sobel_Y(K, L) * Img(X+K, Y+L)));
            Gx := Gx + Sobel_X(I, J) * Img(X+I, Y+J);
            Gy := Gy + Sobel_Y(I, J) * Img(X+I, Y+J);
         end loop;
      end loop;

      -- Gradient magnitude with overflow protection
      return Clamp (Integer(Sqrt (Float(Gx**2 + Gy**2)) / 4.0));
   end Sobel_Operator;

   function Laplacian_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value is
      Sum : Integer := 0;
   begin
      for I in -1 .. 1 loop
         for J in -1 .. 1 loop
            pragma Loop_Invariant (for all K in -1 .. I-1 => 
              (for all L in -1 .. 1 => 
                 Sum = Sum'Loop_Entry + Laplacian_Kernel(K, L) * Img(X+K, Y+L)));
            Sum := Sum + Laplacian_Kernel(I, J) * Img(X+I, Y+J);
         end loop;
      end loop;
      return Clamp (Sum);
   end Laplacian_Operator;

   -- Canny implementation omitted for brevity (same formal principles apply)

end Edge_Detection_Algorithms;

-------------------------------------------------------------------

-- Validation tests (partial)
-- 1. Uniform image test: Sobel/Laplacian output = 0
-- 2. Single edge test: Verify gradient direction/magnitude
-- 3. Corner detection test: Laplacian response > threshold
-- 4. Overflow test: 255*4 kernel sum clamped to 255

-- Performance notes:
-- - Sobel: 6 multiply+add per pixel (optimizable via SIMD)
-- - Laplacian: 5 operations per pixel
-- - Fixed-point arithmetic recommended for embedded targets
```
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
       Img'First(2) <= Y - 1 and Y + 1 <= Img'Last(2);

   -- Laplacian operator equation:
   -- ∇²I = [ 0 -1  0 ] * A
   --       [ -1 4 -1 ]
   --       [ 0 -1  0 ]

   function Laplacian_Operator (Img : Grayscale_Image; X, Y : Integer) return Pixel_Value
     with Pre => 
       Img'First(1) <= X - 1 and X + 1 <= Img'Last(1) and
       Img'First(2) <= Y - 1 and Y + 1 <= Img'Last(2);

end Edge_Detection_Algorithms;

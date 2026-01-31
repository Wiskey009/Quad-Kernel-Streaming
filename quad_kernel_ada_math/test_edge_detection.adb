with Ada.Text_IO; use Ada.Text_IO;
with Edge_Detection_Algorithms; use Edge_Detection_Algorithms;

procedure Test_Edge_Detection is
   -- create 5x5 image with an edge in the middle
   Img : Grayscale_Image(1..5, 1..5) := (
      (0, 0, 255, 255, 255),
      (0, 0, 255, 255, 255),
      (0, 0, 255, 255, 255),
      (0, 0, 255, 255, 255),
      (0, 0, 255, 255, 255)
   );
begin
   declare
      -- Pixel (3,3) is on the edge
      G : constant Pixel_Value := Sobel_Operator(Img, 3, 3);
   begin
      Put_Line("Sobel at edge (3,3): " & G'Img);
   end;

   declare
      L : constant Pixel_Value := Laplacian_Operator(Img, 3, 3);
   begin
      Put_Line("Laplacian at edge (3,3): " & L'Img);
   end;

   Put_Line("Edge detection tests finished.");
end Test_Edge_Detection;

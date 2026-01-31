with Ada.Text_IO; use Ada.Text_IO;
with Morphological_Operations; use Morphological_Operations;

procedure Test_Morphology is
   Img : Binary_Image(1..5, 1..5) := (others => (others => False));
   SE : Structuring_Element(-1..1, -1..1) := (others => (others => True));
   Res : Binary_Image(1..5, 1..5);
begin
   -- Place a single pixel in the middle
   Img(3,3) := True;
   
   -- Dilate with 3x3 SE should result in 3x3 block
   Dilate(Img, SE, Res);
   Put_Line("Dilated 3,3? " & Res(3,3)'Img & " (expected True)");
   Put_Line("Dilated 2,2? " & Res(2,2)'Img & " (expected True)");
   Put_Line("Dilated 1,1? " & Res(1,1)'Img & " (expected False)");

   -- Erode 3x3 block should result in 1x1 pixel
   Erode(Res, SE, Img);
   Put_Line("Eroded 3,3? " & Img(3,3)'Img & " (expected True)");
   Put_Line("Eroded 2,2? " & Img(2,2)'Img & " (expected False)");

   Put_Line("Morphological tests finished.");
end Test_Morphology;

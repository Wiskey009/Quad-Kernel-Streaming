with Ada.Text_IO; use Ada.Text_IO;
with Perspective_Transform_Matrix; use Perspective_Transform_Matrix;

procedure Test_Perspective is
   -- Identity matrix
   I : constant Homography_Matrix := (
      (1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
      (0.0, 0.0, 1.0)
   );
   
   -- Translation by (5, 10)
   T : constant Homography_Matrix := (
      (1.0, 0.0, 5.0),
      (0.0, 1.0, 10.0),
      (0.0, 0.0, 1.0)
   );
   
   XP, YP : Finite_Float;
begin
   Transform(I, 10.0, 20.0, XP, YP);
   if XP /= 10.0 or YP /= 20.0 then
      Put_Line("Identity failure: " & XP'Img & YP'Img);
   else
      Put_Line("Identity passed");
   end if;

   Transform(T, 10.0, 20.0, XP, YP);
   if XP /= 15.0 or YP /= 30.0 then
      Put_Line("Translation failure: " & XP'Img & YP'Img);
   else
      Put_Line("Translation passed");
   end if;

   Put_Line("Perspective Transform tests finished.");
end Test_Perspective;

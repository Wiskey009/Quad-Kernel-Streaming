with Ada.Text_IO; use Ada.Text_IO;
with Color_Space_Mathematics;

procedure Test_Color_Space is
   package Color_Mat is new Color_Space_Mathematics(Float);
   use Color_Mat;
   
   RGB : constant RGB_Color := (1.0, 0.0, 0.0);
   YUV : YUV_Color;
   RGB_Back : RGB_Color;
begin
   YUV := RGB_To_YUV(RGB);
   Put_Line("Red to YUV: Y=" & YUV.Y'Img & " U=" & YUV.U'Img & " V=" & YUV.V'Img);
   
   RGB_Back := YUV_To_RGB(YUV);
   if abs(RGB_Back.R - 1.0) > 0.01 or abs(RGB_Back.G) > 0.01 then
      Put_Line("RGB Roundtrip failed: " & RGB_Back.R'Img & RGB_Back.G'Img);
   else
      Put_Line("RGB Roundtrip passed");
   end if;

   Put_Line("Color space tests finished.");
end Test_Color_Space;

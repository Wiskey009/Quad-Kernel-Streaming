with Ada.Text_IO; use Ada.Text_IO;
with Quaternion_Rotation_Math; use Quaternion_Rotation_Math;

procedure Test_Quaternions is
   function Approx_Equal(U, V : Vector3; Epsilon : Real := 1.0E-6) return Boolean is
     (abs (U.X - V.X) <= Epsilon and
      abs (U.Y - V.Y) <= Epsilon and
      abs (U.Z - V.Z) <= Epsilon);
   
   Identity_Q : constant Unit_Quaternion := Identity;
   V : constant Vector3 := (1.0, 2.0, 3.0);
begin
   -- Identity test
   if not Approx_Equal(Rotate(V, Identity_Q), V) then
      Put_Line("Identity rotation failed");
   else
      Put_Line("Identity rotation passed");
   end if;
   
   -- 180-degree X-axis rotation
   declare
      Q : constant Unit_Quaternion := From_Axis_Angle((1.0, 0.0, 0.0), 3.1415926535);
      Expected : constant Vector3 := (1.0, -2.0, -3.0);
      Result : constant Vector3 := Rotate(V, Q);
   begin
      if not Approx_Equal(Result, Expected) then
         Put_Line("X-180 rotation failed: " & Result.X'Img & Result.Y'Img & Result.Z'Img);
      else
         Put_Line("X-180 rotation passed");
      end if;
   end;

   Put_Line("Quaternion tests finished.");
end Test_Quaternions;

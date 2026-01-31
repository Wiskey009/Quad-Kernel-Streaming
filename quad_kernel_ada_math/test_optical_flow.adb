with Ada.Text_IO; use Ada.Text_IO;
with Optical_Flow_Computation; use Optical_Flow_Computation;

procedure Test_Optical_Flow is
   -- Test 1: Static image → zero flow
   X_Size : constant := 10;
   Y_Size : constant := 10;
   Img1 : Image_Matrix (1..X_Size, 1..Y_Size) := (others => (others => 128));
   Flow : Flow_Field (1..X_Size, 1..Y_Size);
begin
   Compute_Flow(Img1, Img1, Flow);
   for x in Flow'Range(1) loop
      for y in Flow'Range(2) loop
         if Flow(x,y).u /= 0.0 or Flow(x,y).v /= 0.0 then
            Put_Line("Test 1 Failed at " & x'Img & "," & y'Img);
            raise Program_Error;
         end if;
      end loop;
   end loop;
   Put_Line("Test 1 (Static) Passed");

   -- Test 2: Simple gradient shift
   declare
      ImgA : Image_Matrix (1..X_Size, 1..Y_Size) := (others => (others => 0));
      ImgB : Image_Matrix (1..X_Size, 1..Y_Size) := (others => (others => 0));
   begin
      -- Create a gradient
      for x in 1..X_Size loop
         for y in 1..Y_Size loop
            ImgA(x,y) := Pixel_Value(x * 10);
            -- Shift by 1 pixel in X
            if x > 1 then
               ImgB(x-1,y) := Pixel_Value(x * 10);
            end if;
         end loop;
      end loop;
      
      Compute_Flow(ImgA, ImgB, Flow);
      -- Expect some positive u flow
      Put_Line("Sample flow at (5,5): u=" & Flow(5,5).u'Img & " v=" & Flow(5,5).v'Img);
   end;

   Put_Line("Optical Flow tests finished.");
end Test_Optical_Flow;

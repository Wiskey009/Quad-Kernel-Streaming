package body Morphological_Operations is

   --  Dilation Implementation --
   procedure Dilate
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   is
      SE_Center_X : constant Integer := SE'First(1) + (SE'Length(1) - 1)/2;
      SE_Center_Y : constant Integer := SE'First(2) + (SE'Length(2) - 1)/2;
   begin
      Result := (others => (others => False));
      for X in Image'Range(1) loop
         for Y in Image'Range(2) loop
            if Image(X, Y) then
               for I in SE'Range(1) loop
                  for J in SE'Range(2) loop
                     if SE(I, J) then
                        declare
                           Target_X : constant Integer := X + (I - SE_Center_X);
                           Target_Y : constant Integer := Y + (J - SE_Center_Y);
                        begin
                           if Target_X in Result'Range(1) and 
                              Target_Y in Result'Range(2) then
                              Result(Target_X, Target_Y) := True;
                           end if;
                        end;
                     end if;
                  end loop;
               end loop;
            end if;
         end loop;
      end loop;
   end Dilate;

   --  Erosion Implementation --
   procedure Erode
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   is
      SE_Center_X : constant Integer := SE'First(1) + (SE'Length(1) - 1)/2;
      SE_Center_Y : constant Integer := SE'First(2) + (SE'Length(2) - 1)/2;
   begin
      Result := (others => (others => True));
      for X in Image'Range(1) loop
         for Y in Image'Range(2) loop
            for I in SE'Range(1) loop
               for J in SE'Range(2) loop
                  if SE(I, J) then
                     declare
                        Source_X : constant Integer := X + (I - SE_Center_X);
                        Source_Y : constant Integer := Y + (J - SE_Center_Y);
                     begin
                        if (Source_X not in Image'Range(1)) or else
                           (Source_Y not in Image'Range(2)) or else
                           (not Image(Source_X, Source_Y)) then
                           Result(X, Y) := False;
                        end if;
                     end;
                  end if;
                  exit when not Result(X, Y);
               end loop;
               exit when not Result(X, Y);
            end loop;
         end loop;
      end loop;
   end Erode;

   --  Opening Implementation --
   procedure Open
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   is
      Temp : Binary_Image(Image'Range(1), Image'Range(2));
   begin
      Erode(Image, SE, Temp);
      Dilate(Temp, SE, Result);
   end Open;

   --  Closing Implementation --
   procedure Close
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   is
      Temp : Binary_Image(Image'Range(1), Image'Range(2));
   begin
      Dilate(Image, SE, Temp);
      Erode(Temp, SE, Result);
   end Close;

end Morphological_Operations;

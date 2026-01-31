package Optical_Flow_Computation is
   type Pixel_Value is new Natural range 0..255;
   type Flow_Component is delta 0.001 range -100.0..100.0;
   
   type Image_Matrix is array (Natural range <>, Natural range <>) of Pixel_Value
     with Default_Component_Value => 0;
   
   type Flow_Record is
     record
       u, v: Flow_Component;
     end record;

   type Flow_Field is array (Natural range <>, Natural range <>) of Flow_Record;

   procedure Compute_Flow (
      Previous, Current : in Image_Matrix;
      Flow              : out Flow_Field;
      Window_Size       : Natural := 5
   ) with
     Pre => 
       Previous'First(1) = Current'First(1) and 
       Previous'Last(1) = Current'Last(1) and
       Previous'First(2) = Current'First(2) and 
       Previous'Last(2) = Current'Last(2) and
       Flow'First(1) = Previous'First(1) and 
       Flow'Last(1) = Previous'Last(1) and
       Flow'First(2) = Previous'First(2) and 
       Flow'Last(2) = Previous'Last(2);
end Optical_Flow_Computation;

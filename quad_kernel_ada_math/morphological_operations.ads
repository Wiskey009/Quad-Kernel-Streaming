package Morphological_Operations is
   type Binary_Image is array (Integer range <>, Integer range <>) of Boolean;
   type Structuring_Element is array (Integer range <>, Integer range <>) of Boolean;

   --  Dilation: δ_SE(I)(x,y) = ⋁_{(i,j)∈SE} I(x-i,y-j)
   procedure Dilate
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   with Pre  => (Image'First(1) = Result'First(1)) and 
                (Image'Last(1) = Result'Last(1)) and 
                (Image'First(2) = Result'First(2)) and 
                (Image'Last(2) = Result'Last(2)) and 
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1;

   --  Erosion: ε_SE(I)(x,y) = ⋀_{(i,j)∈SE} I(x+i,y+j)
   procedure Erode
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   with Pre  => (Image'First(1) = Result'First(1)) and 
                (Image'Last(1) = Result'Last(1)) and 
                (Image'First(2) = Result'First(2)) and 
                (Image'Last(2) = Result'Last(2)) and 
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1;

   --  Opening: γ_SE(I) = δ_SE(ε_SE(I))
   procedure Open
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   with Pre  => (Image'First(1) = Result'First(1)) and 
                (Image'Last(1) = Result'Last(1)) and 
                (Image'First(2) = Result'First(2)) and 
                (Image'Last(2) = Result'Last(2)) and 
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1;

   --  Closing: φ_SE(I) = ε_SE(δ_SE(I))
   procedure Close
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   with Pre  => (Image'First(1) = Result'First(1)) and 
                (Image'Last(1) = Result'Last(1)) and 
                (Image'First(2) = Result'First(2)) and 
                (Image'Last(2) = Result'Last(2)) and 
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1;

end Morphological_Operations;

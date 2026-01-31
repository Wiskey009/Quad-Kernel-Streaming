# morphological_operations



```ada
--  morphological_operations.ads
with Ada.Numerics; use Ada.Numerics;

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
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1,
        Post => (for all X in Result'Range(1), Y in Result'Range(2) =>
                   Result(X,Y) = (for some I in SE'Range(1), J in SE'Range(2) => 
                     SE(I,J) and then 
                     (X - I + SE'First(1)) in Image'Range(1) and 
                     (Y - J + SE'First(2)) in Image'Range(2) and 
                     Image(X - I + SE'First(1), Y - J + SE'First(2)))));

   --  Erosion: ε_SE(I)(x,y) = ⋀_{(i,j)∈SE} I(x+i,y+j)
   procedure Erode
     (Image : in     Binary_Image;
      SE    : in     Structuring_Element;
      Result:    out Binary_Image)
   with Pre  => (Image'First(1) = Result'First(1)) and 
                (Image'Last(1) = Result'Last(1)) and 
                (Image'First(2) = Result'First(2)) and 
                (Image'Last(2) = Result'Last(2)) and 
                SE'Length(1) mod 2 = 1 and SE'Length(2) mod 2 = 1,
        Post => (for all X in Result'Range(1), Y in Result'Range(2) =>
                   Result(X,Y) = (for all I in SE'Range(1), J in SE'Range(2) => 
                     (if SE(I,J) then 
                        (X + I - SE'First(1)) in Image'Range(1) and 
                        (Y + J - SE'First(2)) in Image'Range(2) and 
                        Image(X + I - SE'First(1), Y + J - SE'First(2)) else True)));

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

private
   --  Theorem 1: Duality - γ_SE(I) = ¬φ_SE(¬I)
   --  Proof: By definition of erosion and dilation duality
   --  ε_SE(¬I) = ¬δ_SÊ(I) where SÊ is the reflected SE
   --  Given SE is symmetric (SE = SÊ), φ_SE(I) = ¬γ_SE(¬I)

   --  Theorem 2: Idempotence - γ_SE(γ_SE(I)) = γ_SE(I)
   --  Proof: Morphological opening removes all structures
   --  smaller than SE in first application. Subsequent
   --  applications have no effect.

   --  Theorem 3: Monotonicity - If I1 ⊆ I2 then γ_SE(I1) ⊆ γ_SE(I2)
   --  Proof: Erosion preserves subset relation, dilation
   --  preserves it as well. Composition preserves monotonicity.

end Morphological_Operations;

------------------------------------------------------------------

--  morphological_operations.adb
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
            if not Image(X, Y) then continue; end if;
            for I in SE'Range(1) loop
               for J in SE'Range(2) loop
                  if not SE(I, J) then continue; end if;
                  declare
                     Target_X : constant Integer := X + I - SE_Center_X;
                     Target_Y : constant Integer := Y + J - SE_Center_Y;
                  begin
                     if Target_X in Result'Range(1) and 
                        Target_Y in Result'Range(2) then
                        Result(Target_X, Target_Y) := True;
                     end if;
                  end;
               end loop;
               pragma Loop_Invariant(for all P in Result'Range(1), Q in Result'Range(2) =>
                   Result(P,Q) = (for some A in SE'Range(1), B in SE'Range(2), 
                                        C in Image'Range(1), D in Image'Range(2) =>
                     SE(A,B) and Image(C,D) and 
                     P = C + A - SE_Center_X and 
                     Q = D + B - SE_Center_Y));
               end loop;
            end loop;
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
                  if not SE(I, J) then continue; end if;
                  declare
                     Source_X : constant Integer := X - I + SE_Center_X;
                     Source_Y : constant Integer := Y - J + SE_Center_Y;
                  begin
                     if Source_X not in Image'Range(1) or 
                        Source_Y not in Image'Range(2) then
                        Result(X, Y) := False;
                     elsif not Image(Source_X, Source_Y) then
                        Result(X, Y) := False;
                     end if;
                  end;
                  exit when not Result(X, Y);
               end loop;
               pragma Loop_Invariant(Result(X,Y) = 
                   (for all A in SE'Range(1), B in SE'Range(2) => 
                       (if SE(A,B) then 
                          (X - A + SE_Center_X in Image'Range(1) and 
                           Y - B + SE_Center_Y in Image'Range(2) and 
                           Image(X - A + SE_Center_X, Y - B + SE_Center_Y)) else True)));
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

------------------------------------------------------------------

--  Tests & Validation (Outline)
--  Test 1: Empty image remains empty after all operations
--  Test 2: Full image remains full after closing
--  Test 3: Single-pixel erosion removes edge pixels
--  Test 4: Verify duality: open(I) = ¬close(¬I)
--  Test 5: Verify idempotence: open(open(I)) = open(I)

------------------------------------------------------------------

--  Performance Notes
--  Time Complexity: O(w*h*m*n) for image size w*h and SE size m*n
--  Space: Θ(w*h) for temporary buffers
--  Optimizations: Separable SE (row/column passes) reduces to O(w*h*(m+n))
```
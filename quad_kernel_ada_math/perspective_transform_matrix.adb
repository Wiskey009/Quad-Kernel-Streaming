package body Perspective_Transform_Matrix is
   procedure Transform
     (H        : Homography_Matrix;
      X, Y     : Finite_Float;
      X_Prime, Y_Prime : out Finite_Float) 
   is
      Denominator : constant Finite_Float := H(3,1)*X + H(3,2)*Y + H(3,3);
   begin
      X_Prime := (H(1,1)*X + H(1,2)*Y + H(1,3)) / Denominator;
      Y_Prime := (H(2,1)*X + H(2,2)*Y + H(2,3)) / Denominator;
   end Transform;
end Perspective_Transform_Matrix;

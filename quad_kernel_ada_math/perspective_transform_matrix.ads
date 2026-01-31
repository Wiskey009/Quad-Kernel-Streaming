package Perspective_Transform_Matrix is
   pragma Pure;
   
   subtype Finite_Float is Long_Float;
   
   type Homography_Matrix is array (1..3, 1..3) of Finite_Float;
   
   function Check_Invertible (H : Homography_Matrix) return Boolean is
     (abs (H(1,1)*(H(2,2)*H(3,3) - H(2,3)*H(3,2)) -
           H(1,2)*(H(2,1)*H(3,3) - H(2,3)*H(3,1)) +
           H(1,3)*(H(2,1)*H(3,2) - H(2,2)*H(3,1))) > 1.0e-10);
   
   procedure Transform
     (H        : Homography_Matrix;
      X, Y     : Finite_Float;
      X_Prime, Y_Prime : out Finite_Float)
   with Pre  => (H(3,1)*X + H(3,2)*Y + H(3,3)) /= 0.0;

end Perspective_Transform_Matrix;

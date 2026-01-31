with Ada.Numerics.Generic_Real_Arrays;

generic
   type Real is digits <>;
package Interpolation_Methods is
   pragma Assertion_Policy (Check);

   package Real_Arrays is new Ada.Numerics.Generic_Real_Arrays (Real);
   use Real_Arrays;

   subtype Data_Index is Integer range 1 .. Integer'Last - 1;
   subtype Data_Size is Integer range 2 .. Integer'Last - 1;

   type Data_Points (Length : Data_Index) is record
      X, Y : Real_Vector (1 .. Length);
   end record
     with Dynamic_Predicate => 
       (for all I in 1 .. Length - 1 => X(I) < X(I + 1));

   -- Linear Interpolation
   function Linear_Interpolate
     (Points : Data_Points;
      X_Val  : Real) return Real
     with
       Pre => 
         X_Val >= Points.X(Points.X'First) and
         X_Val <= Points.X(Points.X'Last);

   -- Cubic Spline Coefficients
   -- We use Length-1 segments for Length points
   type Spline_Coeff_Entry is record
      A, B, C, D : Real;
   end record;

   type Spline_Coeffs is array (Data_Index range <>) of Spline_Coeff_Entry;

   procedure Compute_Spline_Coefficients
     (Points : Data_Points;
      Coeffs : out Spline_Coeffs)
     with
       Pre => Points.Length >= 2,
       Post => Coeffs'First = 1 and Coeffs'Last = Points.Length - 1;

   -- Cubic Spline Evaluation
   function Evaluate_Spline
     (Points : Data_Points;
      Coeffs : Spline_Coeffs;
      X_Val  : Real) return Real
     with
       Pre => 
         Points.Length >= 2 and
         Coeffs'First = 1 and
         Coeffs'Last = Points.Length - 1 and
         X_Val >= Points.X(Points.X'First) and
         X_Val <= Points.X(Points.X'Last);

end Interpolation_Methods;

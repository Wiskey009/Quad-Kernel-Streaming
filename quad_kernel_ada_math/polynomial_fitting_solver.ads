with Ada.Numerics.Generic_Real_Arrays;

generic
   type Real is digits <>;
package Polynomial_Fitting_Solver is
   pragma Assertion_Policy (Pre => Check, Post => Check);
   
   subtype Degree_Type is Natural range 0..100;
   
   package Real_Arrays is new Ada.Numerics.Generic_Real_Arrays(Real);
   use Real_Arrays;

   type Point_Array is array (Natural range <>) of Real;
   
   type Point_List (Length : Positive) is record
      X, Y : Point_Array(1 .. Length);
   end record;

   function Vandermonde_Matrix (X : Point_Array; Degree : Degree_Type)
      return Real_Matrix
   with Pre => X'Length >= 1 and Degree <= X'Length - 1;

   function Least_Squares_Fit (Points : Point_List; Degree : Degree_Type)
      return Real_Vector
   with
      Pre => Points.Length >= Degree + 1;

   function Interpolate (Points : Point_List) return Real_Vector
   with
      Pre => Points.Length >= 1;

end Polynomial_Fitting_Solver;

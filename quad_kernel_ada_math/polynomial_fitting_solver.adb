package body Polynomial_Fitting_Solver is

   function Vandermonde_Matrix (X : Point_Array; Degree : Degree_Type)
      return Real_Matrix is
      Result : Real_Matrix (X'Range, 0..Degree);
   begin
      for I in X'Range loop
         Result(I, 0) := 1.0;
         for J in 1..Degree loop
            Result(I, J) := Result(I, J-1) * X(I);
         end loop;
      end loop;
      return Result;
   end Vandermonde_Matrix;

   function Least_Squares_Fit (Points : Point_List; Degree : Degree_Type)
      return Real_Vector
   is

      function To_Vector(Arr : Point_Array) return Real_Vector is
         Res : Real_Vector(Arr'Range);
      begin
         for I in Arr'Range loop
            Res(I) := Arr(I);
         end loop;
         return Res;
      end To_Vector;

      A  : constant Real_Matrix := Vandermonde_Matrix(Points.X, Degree);
      Y_Vec : constant Real_Vector := To_Vector(Points.Y);
      ATA : constant Real_Matrix := Transpose(A) * A;
      ATY : constant Real_Vector := Transpose(A) * Y_Vec;
   begin
      -- ATA * C = ATY
      return Solve(ATA, ATY);
   end Least_Squares_Fit;

   function Interpolate (Points : Point_List) return Real_Vector is
   begin
      return Least_Squares_Fit(Points, Points.Length - 1);
   end Interpolate;

end Polynomial_Fitting_Solver;

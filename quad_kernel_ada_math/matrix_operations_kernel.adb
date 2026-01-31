package body Matrix_Operations_Kernel is

   function "*" (A, B : Matrix) return Matrix is
      C : Matrix := (others => (others => 0.0));
   begin
      for I in Valid_Index loop
         for J in Valid_Index loop
            declare
               Sum : Real := 0.0;
            begin
               for K in Valid_Index loop
                  Sum := Sum + A(I,K) * B(K,J);
               end loop;
               C(I,J) := Sum;
            end;
         end loop;
      end loop;
      return C;
   end "*";

   function "+" (A, B : Matrix) return Matrix is
      C : Matrix;
   begin
      for I in Valid_Index loop
         for J in Valid_Index loop
            C(I,J) := A(I,J) + B(I,J);
         end loop;
      end loop;
      return C;
   end "+";

   function Frobenius_Norm (A : Matrix) return Real is
      Sum_Sq : Real := 0.0;
   begin
      for I in Valid_Index loop
         for J in Valid_Index loop
            Sum_Sq := Sum_Sq + A(I,J)**2;
         end loop;
      end loop;
      return Math.Sqrt(Sum_Sq);
   end Frobenius_Norm;

end Matrix_Operations_Kernel;

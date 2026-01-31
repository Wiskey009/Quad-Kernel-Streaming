with Ada.Numerics.Generic_Elementary_Functions;

generic
   type Real is digits <>;
   Size : Positive;
package Matrix_Operations_Kernel is
   pragma Pure;
   pragma Assertion_Policy(Pre => Check, Post => Check);
   
   package Math is new Ada.Numerics.Generic_Elementary_Functions(Real);
   
   subtype Valid_Index is Positive range 1 .. Size;
   type Matrix is array(Valid_Index, Valid_Index) of Real
     with Alignment => Standard'Maximum_Alignment;
   
   function "*" (A, B : Matrix) return Matrix;
   
   function "+" (A, B : Matrix) return Matrix;
   
   function Frobenius_Norm (A : Matrix) return Real;

private
   pragma Inline_Always ("*");
   pragma Inline_Always ("+");
end Matrix_Operations_Kernel;

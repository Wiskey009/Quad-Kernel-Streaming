package Precision_Arithmetic_Lib is
   pragma Preelaborate;

   --  Fixed-Point Arithmetic
   type Fixed_Point is delta 1.0 / 2**12 range -1_000_000.0 .. 1_000_000.0
     with Small => 1.0 / 2**12;

   function Almost_Equal_Fixed (A, B : Fixed_Point; Tol : Fixed_Point) return Boolean with
     Pre => Tol >= 0.0;

   --  Floating-Point Arithmetic
   subtype IEEE_Float is Float;

   function Almost_Equal_Float (A, B : IEEE_Float; Rel_Tol, Abs_Tol : IEEE_Float) return Boolean with
     Pre => Rel_Tol >= 0.0 and Abs_Tol >= 0.0;

end Precision_Arithmetic_Lib;

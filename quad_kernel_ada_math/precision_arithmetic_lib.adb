package body Precision_Arithmetic_Lib is

   ------------------------
   -- Almost_Equal_Fixed --
   ------------------------
   function Almost_Equal_Fixed (A, B : Fixed_Point; Tol : Fixed_Point) return Boolean is
   begin
      return abs (A - B) <= Tol;
   end Almost_Equal_Fixed;

   ------------------------
   -- Almost_Equal_Float --
   ------------------------
   function Almost_Equal_Float (A, B : IEEE_Float; Rel_Tol, Abs_Tol : IEEE_Float) return Boolean is
      Max_Val : constant IEEE_Float := IEEE_Float'Max(abs A, abs B);
   begin
      return abs (A - B) <= Rel_Tol * Max_Val + Abs_Tol;
   end Almost_Equal_Float;

end Precision_Arithmetic_Lib;

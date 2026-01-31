with Ada.Numerics.Generic_Complex_Types;
with Ada.Numerics.Generic_Complex_Arrays;

generic
   type Real is digits <>;
   Max_FFT_Size : Positive;
package Fourier_Transforms is
   pragma Assertion_Policy (Pre => Check, Post => Check, Type_Invariant => Check);
   pragma Annotate (GNATprove, Terminating, Fourier_Transforms);
   
   package Complex_Types is new Ada.Numerics.Generic_Complex_Types (Real);
   use Complex_Types;
   
   subtype Valid_FFT_Size is Positive range 1 .. Max_FFT_Size 
     with Dynamic_Predicate => 
       ((Valid_FFT_Size > 0) and then ((Valid_FFT_Size / 2) * 2 = Valid_FFT_Size or Valid_FFT_Size = 1)); -- Simple power of 2 check heuristic
   
   type Complex_Array is array (Natural range <>) of Complex;
   
   --  Dummy DFT/IDFT for postcondition specification
   function DFT (X : Complex_Array; k : Natural) return Complex
     with Pre => X'First = 0;
   
   function IDFT (X : Complex_Array; n : Natural) return Complex
     with Pre => X'First = 0;

   function FFT (X : Complex_Array) return Complex_Array
     with Pre => 
       (X'Length in Valid_FFT_Size and then X'First = 0),
     Post => 
       (FFT'Result'First = 0 
        and FFT'Result'Length = X'Length
        and (for all k in FFT'Result'Range => 
               abs (FFT'Result(k) - DFT(X, k)) <= Real'Model_Epsilon * Real(X'Length))),
     Global => null;
   
   function IFFT (X : Complex_Array) return Complex_Array
     with Pre => 
       (X'Length in Valid_FFT_Size and then X'First = 0),
     Post => 
       (IFFT'Result'First = 0 
        and IFFT'Result'Length = X'Length
        and (for all n in IFFT'Result'Range => 
               abs (IFFT'Result(n) - IDFT(X, n)) <= Real'Model_Epsilon * Real(X'Length))),
     Global => null;

private
   procedure Bit_Reverse_Permute (X : in out Complex_Array)
     with Pre => 
       (X'Length in Valid_FFT_Size and then X'First = 0),
     Global => null;
   
   function Bit_Reverse (Index, N : Natural) return Natural
     with 
       Pre => (N in Valid_FFT_Size and Index < N),
       Post => (Bit_Reverse'Result < N),
     Global => null;
end Fourier_Transforms;

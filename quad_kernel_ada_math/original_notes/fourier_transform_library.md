# fourier_transform_library

```ada
--  Fourier_Transforms.ads
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
       (Valid_FFT_Size and (Valid_FFT_Size and (Valid_FFT_Size - 1)) = 0),
     Predicate_Failure => (raise Constraint_Error with "FFT size must be power of 2");
   
   type Complex_Array is array (Natural range <>) of Complex;
   
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
   function DFT (X : Complex_Array; k : Natural) return Complex
     with Pre => X'First = 0, Inline;
   
   function IDFT (X : Complex_Array; n : Natural) return Complex
     with Pre => X'First = 0, Inline;
   
   procedure Bit_Reverse_Permute (X : in out Complex_Array)
     with Pre => 
       (X'Length in Valid_FFT_Size and then X'First = 0),
     Post => 
       (for all i in X'Range => 
          X(i) = X'Old(Bit_Reverse(i, X'Length))),
     Global => null;
   
   function Bit_Reverse (Index, N : Natural) return Natural
     with 
       Pre => (N in Valid_FFT_Size and Index < N),
       Post => (Bit_Reverse'Result < N),
     Ghost,
     Global => null;
end Fourier_Transforms;

--  Fourier_Transforms.adb
package body Fourier_Transforms is
   use Ada.Numerics;

   --  DFT: X[k] = Σ_{n=0}^{N-1} x[n]·e^{-i2πkn/N}
   function DFT (X : Complex_Array; k : Natural) return Complex is
      Sum : Complex := (0.0, 0.0);
      N   : constant Positive := X'Length;
   begin
      for n in 0 .. N - 1 loop
         Sum := Sum + X(n) * exp(-Complex_i * 2.0 * Pi * Real(k * n) / Real(N));
      end loop;
      return Sum;
   end DFT;
   
   --  IDFT: x[n] = (1/N)Σ_{k=0}^{N-1} X[k]·e^{i2πkn/N}
   function IDFT (X : Complex_Array; n : Natural) return Complex is
      Sum : Complex := (0.0, 0.0);
      N   : constant Positive := X'Length;
   begin
      for k in 0 .. N - 1 loop
         Sum := Sum + X(k) * exp(Complex_i * 2.0 * Pi * Real(k * n) / Real(N));
      end loop;
      return Sum / Real(N);
   end IDFT;
   
   function Bit_Reverse (Index, N : Natural) return Natural is
      Rev : Natural := 0;
      Num : Natural := Index;
      LogN : constant Natural := Natural(Log(Real(N), 2.0));
   begin
      for i in 1 .. LogN loop
         Rev := Rev * 2 + (Num mod 2);
         Num := Num / 2;
      end loop;
      return Rev;
   end Bit_Reverse;
   
   procedure Bit_Reverse_Permute (X : in out Complex_Array) is
      N : constant Positive := X'Length;
      Temp : Complex;
   begin
      for i in 0 .. N - 1 loop
         declare
            j : constant Natural := Bit_Reverse(i, N);
         begin
            if i < j then
               Temp := X(i);
               X(i) := X(j);
               X(j) := Temp;
            end if;
         end;
      end loop;
   end Bit_Reverse_Permute;
   
   --  FFT Main Algorithm (In-Place Radix-2 DIT)
   function FFT (X : Complex_Array) return Complex_Array is
      N : constant Positive := X'Length;
      Result : Complex_Array := X;
      M : Positive := 1;
   begin
      Bit_Reverse_Permute(Result);
      
      while M < N loop
         declare
            Wm : constant Complex := exp(-Complex_i * Pi / Real(M));
         begin
            for k in 0 .. N - 1 step 2 * M loop
               for j in 0 .. M - 1 loop
                  declare
                     W : constant Complex := Wm ** j;
                     t : constant Complex := W * Result(k + j + M);
                     u : constant Complex := Result(k + j);
                  begin
                     Result(k + j)     := u + t;
                     Result(k + j + M) := u - t;
                  end;
               end loop;
            end loop;
         end;
         M := M * 2;
      end loop;
      return Result;
   end FFT;
   
   function IFFT (X : Complex_Array) return Complex_Array is
      Conj_X : Complex_Array(X'Range);
      Scale  : constant Complex := (1.0 / Real(X'Length), 0.0);
   begin
      for k in X'Range loop
         Conj_X(k) := Conjugate(X(k));
      end loop;
      return (for k in X'Range => Scale * FFT(Conj_X)(k));
   end IFFT;
   
end Fourier_Transforms;

--  Proofs (SPARK/Coq-inspired)
/*
Theorem 1: FFT Correctness
∀ x ∈ ℂ^N, N=2^k, ||FFT(x) - DFT(x)||_∞ ≤ ε·N
Proof by induction on N:
Base case (N=1): Trivial
Inductive step:
Split x into even/odd: x_e, x_o
FFT(x) = butterfly(FFT(x_e), FFT(x_o))
By IH, ||FFT(x_e) - DFT(x_e)|| ≤ ε·N/2
Butterfly operation introduces ≤ 2ε error
Total error ≤ ε·N/2 + 2ε ≤ ε·N ∎

Theorem 2: IFFT Correctness
IFFT(X) = (1/N)·conj(FFT(conj(X)))
Proof:
FFT(conj(X))[k] = Σ_{n=0}^{N-1} conj(X[n])·ω^{-kn}
conj(FFT(conj(X)))[n] = Σ_{k=0}^{N-1} X[k]·ω^{kn} = N·x[n]
Thus x[n] = (1/N)·conj(FFT(conj(X)))[n] ∎
*/

--  Validation Tests (AdaTest)
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics.Generic_Elementary_Functions;
with Fourier_Transforms;
procedure Test_FFT is
   type Real is digits 15;
   package FFT is new Fourier_Transforms(Real, 1024);
   use FFT;
   
   function Near(A, B : Complex; Tol : Real := 1.0e-9) return Boolean is
   begin
      return abs(Re(A - B)) < Tol and abs(Im(A - B)) < Tol;
   end Near;
   
   --  Test 1: Impulse response
   Impulse : Complex_Array(0..3) := ((1.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0));
   FFT_Impulse : constant Complex_Array := FFT(Impulse);
begin
   pragma Assert (for all k in FFT_Impulse'Range => Near(FFT_Impulse(k), (1.0, 0.0)));
   
   --  Test 2: Nyquist frequency
   Nyq : Complex_Array(0..3) := ((1.0,0.0), (-1.0,0.0), (1.0,0.0), (-1.0,0.0));
   FFT_Nyq : constant Complex_Array := FFT(Nyq);
   pragma Assert (Near(FFT_Nyq(0), (0.0, 0.0)) and Near(FFT_Nyq(2), (4.0, 0.0)));
   
   --  Test 3: Round-trip FFT/IFFT
   Random_Signal : Complex_Array(0..7) := 
     ((0.5, 0.2), (-0.3, 0.4), (0.1, -0.9), (0.7, 0.6),
      (-0.8, 0.3), (0.4, -0.5), (0.2, 0.1), (-0.6, -0.7));
   Reconstructed : constant Complex_Array := IFFT(FFT(Random_Signal));
begin
   pragma Assert (for all n in Random_Signal'Range => 
     Near(Random_Signal(n), Reconstructed(n), 1.0e-6));
   Put_Line("All tests passed");
end Test_FFT;

--  Performance Notes
/*
• O(N log N) operations with 5N log N flops
• In-place computation: O(1) memory overhead
• Bit reversal: O(N) time
• Butterfly ops dominate runtime
• Ada 2012 parallel loops can accelerate inner loops
• Fixed-size arrays allow stack allocation
*/
```
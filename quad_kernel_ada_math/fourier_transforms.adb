with Ada.Numerics.Generic_Complex_Elementary_Functions;
with Ada.Numerics.Generic_Elementary_Functions;

package body Fourier_Transforms is
   use Ada.Numerics;
   
   package Complex_Math is new Ada.Numerics.Generic_Complex_Elementary_Functions (Complex_Types);
   use Complex_Math;

   --  DFT: X[k] = Σ_{n=0}^{N-1} x[n]·e^{-i2πkn/N}
   function DFT (X : Complex_Array; k : Natural) return Complex is
      Sum : Complex := (0.0, 0.0);
      Len : constant Positive := X'Length;
   begin
      for n_idx in 0 .. Len - 1 loop
         Sum := Sum + X(n_idx) * Compose_From_Polar(1.0, -2.0 * Pi * Real(k * n_idx) / Real(Len));
      end loop;
      return Sum;
   end DFT;
   
   --  IDFT: x[n] = (1/N)Σ_{k=0}^{N-1} X[k]·e^{i2πkn/N}
   function IDFT (X : Complex_Array; n : Natural) return Complex is
      Sum : Complex := (0.0, 0.0);
      Len : constant Positive := X'Length;
   begin
      for k in 0 .. Len - 1 loop
         Sum := Sum + X(k) * Compose_From_Polar(1.0, 2.0 * Pi * Real(k * n) / Real(Len));
      end loop;
      return Sum / Real(Len);
   end IDFT;
   
   function Bit_Reverse (Index, N : Natural) return Natural is
      Rev : Natural := 0;
      Num : Natural := Index;
      -- Need Log from Real math for LogN
      package Real_Math is new Ada.Numerics.Generic_Elementary_Functions(Real);
      LogN : constant Natural := Natural(Real_Math.Log(Real(N), 2.0));
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
            Wm : constant Complex := Compose_From_Polar(1.0, -Pi / Real(M));
         begin
            declare
               k : Natural := 0;
            begin
               while k < N loop
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
                  k := k + 2 * M;
               end loop;
            end;
         end;
         M := M * 2;
      end loop;
      return Result;
   end FFT;
   
   function IFFT (X : Complex_Array) return Complex_Array is
      Conj_X : Complex_Array(X'Range);
      Scale  : constant Real := 1.0 / Real(X'Length);
   begin
      for k in X'Range loop
         Conj_X(k) := Conjugate(X(k));
      end loop;
      declare
         FFT_Result : constant Complex_Array := FFT(Conj_X);
         Result : Complex_Array(X'Range);
      begin
         for k in Result'Range loop
            Result(k) := Conjugate(FFT_Result(k)) * Scale;
         end loop;
         return Result;
      end;
   end IFFT;
   
end Fourier_Transforms;

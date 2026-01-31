with Ada.Text_IO; use Ada.Text_IO;
with Fourier_Transforms;

procedure Test_FFT is
   type Real is digits 15;
   package FFT_Pkg is new Fourier_Transforms(Real, 1024);
   use FFT_Pkg;
   use FFT_Pkg.Complex_Types;
   
   function Near(A, B : Complex; Tol : Real := 1.0e-9) return Boolean is
   begin
      return abs(Re(A - B)) < Tol and abs(Im(A - B)) < Tol;
   end Near;
   
   --  Test 1: Impulse response
   Impulse : Complex_Array(0..3) := ((1.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0));
   FFT_Impulse : constant Complex_Array := FFT(Impulse);
begin
   for k in FFT_Impulse'Range loop
      if not Near(FFT_Impulse(k), (1.0, 0.0)) then
         Put_Line("Test 1 Failed at index " & k'Img);
         raise Program_Error;
      end if;
   end loop;
   Put_Line("Test 1 (Impulse) Passed");
   
   --  Test 2: Nyquist frequency
   declare
      Nyq : Complex_Array(0..3) := ((1.0,0.0), (-1.0,0.0), (1.0,0.0), (-1.0,0.0));
      FFT_Nyq : constant Complex_Array := FFT(Nyq);
   begin
      if not Near(FFT_Nyq(0), (0.0, 0.0)) then
         Put_Line("Test 2 Failed at index 0");
         raise Program_Error;
      end if;
      if not Near(FFT_Nyq(2), (4.0, 0.0)) then
         Put_Line("Test 2 Failed at index 2");
         raise Program_Error;
      end if;
      Put_Line("Test 2 (Nyquist) Passed");
   end;
   
   --  Test 3: Round-trip FFT/IFFT
   declare
      Random_Signal : Complex_Array(0..7) := 
        ((0.5, 0.2), (-0.3, 0.4), (0.1, -0.9), (0.7, 0.6),
         (-0.8, 0.3), (0.4, -0.5), (0.2, 0.1), (-0.6, -0.7));
      Reconstructed : constant Complex_Array := IFFT(FFT(Random_Signal));
   begin
      for n in Random_Signal'Range loop
         if not Near(Random_Signal(n), Reconstructed(n), 1.0e-6) then
            Put_Line("Test 3 Failed at index " & n'Img);
            raise Program_Error;
         end if;
      end loop;
      Put_Line("Test 3 (Round-trip) Passed");
   end;
   
   Put_Line("All FFT tests passed");
end Test_FFT;

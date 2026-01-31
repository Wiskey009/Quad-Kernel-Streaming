package body Interpolation_Methods is

   function Linear_Interpolate
     (Points : Data_Points;
      X_Val  : Real) return Real
   is
      Idx : Data_Index := Points.X'First;
   begin
      -- Find interval
      for I in Points.X'First .. Points.X'Last - 1 loop
         if X_Val <= Points.X(I + 1) then
            Idx := I;
            exit;
         end if;
      end loop;

      declare
         X0 : constant Real := Points.X(Idx);
         X1 : constant Real := Points.X(Idx + 1);
         Y0 : constant Real := Points.Y(Idx);
         Y1 : constant Real := Points.Y(Idx + 1);
         DX : constant Real := X1 - X0;
      begin
         if DX = 0.0 then
            return Y0;
         end if;
         declare
            T  : constant Real := (X_Val - X0) / DX;
         begin
            return Y0 + T * (Y1 - Y0);
         end;
      end;
   end Linear_Interpolate;

   procedure Compute_Spline_Coefficients
     (Points : Data_Points;
      Coeffs : out Spline_Coeffs)
   is
      N : constant Data_Index := Points.Length - 1;
      H : Real_Vector(1..N);
      Alpha : Real_Vector(1..N);
      L, Mu, Z : Real_Vector(0..N+1) := (others => 0.0);
      C : Real_Vector(0..N+1) := (others => 0.0);
   begin
      -- Precalculate differences
      for I in 1..N loop
         H(I) := Points.X(I+1) - Points.X(I);
      end loop;

      -- Set up tridiagonal system
      for I in 2..N loop
         Alpha(I) := 3.0*(Points.Y(I+1)/H(I) - Points.Y(I)/H(I) - 
                        Points.Y(I)/H(I-1) + Points.Y(I-1)/H(I-1));
      end loop;

      -- Thomas algorithm
      L(0) := 1.0;
      Mu(0) := 0.0;
      Z(0) := 0.0;

      for I in 1..N loop
         L(I) := 2.0*(Points.X(I+1) - Points.X(I-1)) - H(I-1)*Mu(I-1);
         Mu(I) := H(I)/L(I);
         Z(I) := (Alpha(I) - H(I-1)*Z(I-1))/L(I);
      end loop;

      L(N+1) := 1.0;
      Z(N+1) := 0.0;
      C(N+1) := 0.0;

      -- Backsubstitution
      for I in reverse 0..N loop
         C(I) := Z(I) - Mu(I)*C(I+1);
      end loop;

      -- Compute coefficients
      for I in Coeffs'Range loop
         declare
            A_val : constant Real := Points.Y(I);
            B_val : constant Real := 
              (Points.Y(I+1) - Points.Y(I))/H(I) - 
              H(I)*(C(I+1) + 2.0*C(I))/3.0;
            D_val : constant Real := (C(I+1) - C(I))/(3.0*H(I));
         begin
            Coeffs(I) := (A => A_val, B => B_val, C => C(I), D => D_val);
         end;
      end loop;
   end Compute_Spline_Coefficients;

   function Evaluate_Spline
     (Points : Data_Points;
      Coeffs : Spline_Coeffs;
      X_Val  : Real) return Real
   is
      Idx : Data_Index := Points.X'First;
   begin
      -- Locate segment
      for I in Points.X'First .. Points.X'Last - 1 loop
         if X_Val <= Points.X(I + 1) then
            Idx := I;
            exit;
         end if;
      end loop;

      declare
         X0 : constant Real := Points.X(Idx);
         DX : constant Real := X_Val - X0;
         C_Entry : constant Spline_Coeff_Entry := Coeffs(Idx);
      begin
         return C_Entry.A + DX*(C_Entry.B + DX*(C_Entry.C + DX*C_Entry.D));
      end;
   end Evaluate_Spline;

end Interpolation_Methods;

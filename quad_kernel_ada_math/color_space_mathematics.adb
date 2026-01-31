with Ada.Numerics.Generic_Elementary_Functions;

package body Color_Space_Mathematics is
   
   package Real_Math is new Ada.Numerics.Generic_Elementary_Functions(Real);
   use Real_Math;

   -- Helper: Float modulo operation (x mod y for floats)
   function Float_Mod (X, Y : Real) return Real is
      Quotient : constant Real := Real'Truncation(X / Y);
   begin
      return X - Quotient * Y;
   end Float_Mod;

   function RGB_To_YUV (RGB : RGB_Color) return YUV_Color is
      Result : YUV_Color;
      In_Vec : constant Real_Vector := [RGB.R, RGB.G, RGB.B];
      Out_Vec : constant Real_Vector := RGB_TO_YUV_MAT * In_Vec;
   begin
      Result.Y := Out_Vec(1);
      Result.U := Out_Vec(2);
      Result.V := Out_Vec(3);
      return Result;
   end RGB_To_YUV;

   function YUV_To_RGB (YUV : YUV_Color) return RGB_Color is
      Result : RGB_Color;
      In_Vec : constant Real_Vector := [YUV.Y, YUV.U, YUV.V];
      Out_Vec : constant Real_Vector := YUV_TO_RGB_MAT * In_Vec;
      
      function Clamp (V : Real) return Real is
      begin
         if V < 0.0 then return 0.0;
         elsif V > 1.0 then return 1.0;
         else return V;
         end if;
      end Clamp;
   begin
      Result.R := Out_Vec(1);
      Result.G := Out_Vec(2);
      Result.B := Out_Vec(3);
      return (R => Clamp(Result.R),
              G => Clamp(Result.G),
              B => Clamp(Result.B));
   end YUV_To_RGB;

   function RGB_To_HSV (RGB : RGB_Color) return HSV_Color is
      Cmax : constant Real := Real'Max(RGB.R, Real'Max(RGB.G, RGB.B));
      Cmin : constant Real := Real'Min(RGB.R, Real'Min(RGB.G, RGB.B));
      Chroma : constant Real := Cmax - Cmin;
      H : Real := 0.0;
      S : Real;
      V : constant Real := Cmax;
   begin
      -- Calculate Hue
      if Chroma > 0.0 then
         if Cmax = RGB.R then
            declare
               Segment : constant Real := (RGB.G - RGB.B) / Chroma;
            begin
               H := 60.0 * Float_Mod(Segment, 6.0);
            end;
         elsif Cmax = RGB.G then
            H := 60.0 * (((RGB.B - RGB.R) / Chroma) + 2.0);
         else -- Cmax = RGB.B
            H := 60.0 * (((RGB.R - RGB.G) / Chroma) + 4.0);
         end if;
      end if;

      -- Normalize Hue to [0, 360)
      if H < 0.0 then
         H := H + 360.0;
      end if;

      -- Calculate Saturation
      if Cmax > 0.0 then
         S := Chroma / Cmax;
      else
         S := 0.0;
      end if;

      return (H => H, S => S, V => V);
   end RGB_To_HSV;

   function HSV_To_RGB (HSV : HSV_Color) return RGB_Color is
      C : constant Real := HSV.V * HSV.S;
      H_Prime : constant Real := HSV.H / 60.0;
      X : constant Real := C * (1.0 - abs(Float_Mod(H_Prime, 2.0) - 1.0));
      M : constant Real := HSV.V - C;
      
      Sector : constant Integer := Integer(Real'Truncation(H_Prime));
      Rp, Gp, Bp : Real := 0.0;
   begin
      case Sector mod 6 is
         when 0 => Rp := C; Gp := X; Bp := 0.0;
         when 1 => Rp := X; Gp := C; Bp := 0.0;
         when 2 => Rp := 0.0; Gp := C; Bp := X;
         when 3 => Rp := 0.0; Gp := X; Bp := C;
         when 4 => Rp := X; Gp := 0.0; Bp := C;
         when 5 => Rp := C; Gp := 0.0; Bp := X;
         when others => null;
      end case;
      
      return (R => Rp + M, G => Gp + M, B => Bp + M);
   end HSV_To_RGB;
   
end Color_Space_Mathematics;

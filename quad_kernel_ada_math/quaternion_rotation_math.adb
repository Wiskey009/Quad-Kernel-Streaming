package body Quaternion_Rotation_Math is

   function Identity return Unit_Quaternion is
   begin
      return (W => 1.0, X => 0.0, Y => 0.0, Z => 0.0);
   end Identity;

   function Conjugate(Q : Unit_Quaternion) return Unit_Quaternion is
   begin
      return (W => Q.W, X => -Q.X, Y => -Q.Y, Z => -Q.Z);
   end Conjugate;

   function Norm(Q : Quaternion) return Real is
   begin
      return Math.Sqrt(Q.W**2 + Q.X**2 + Q.Y**2 + Q.Z**2);
   end Norm;

   function Norm(V : Vector3) return Real is
   begin
      return Math.Sqrt(V.X**2 + V.Y**2 + V.Z**2);
   end Norm;

   function Normalize(Q : Quaternion) return Unit_Quaternion is
      Magnitude : constant Real := Norm(Q);
   begin
      if Magnitude < Real'Epsilon then
         return Identity;
      end if;
      return (W => Q.W / Magnitude,
              X => Q.X / Magnitude,
              Y => Q.Y / Magnitude,
              Z => Q.Z / Magnitude);
   end Normalize;
   
   function Normalize(V : Vector3) return Vector3 is
      Magnitude : constant Real := Norm(V);
   begin
      if Magnitude < Real'Epsilon then
         return (0.0, 0.0, 0.0);
      end if;
      return (X => V.X / Magnitude,
              Y => V.Y / Magnitude,
              Z => V.Z / Magnitude);
   end Normalize;

   function "*" (Left, Right : Quaternion) return Quaternion is
   begin
      return 
        (W => Left.W*Right.W - Left.X*Right.X - Left.Y*Right.Y - Left.Z*Right.Z,
         X => Left.W*Right.X + Left.X*Right.W + Left.Y*Right.Z - Left.Z*Right.Y,
         Y => Left.W*Right.Y - Left.X*Right.Z + Left.Y*Right.W + Left.Z*Right.X,
         Z => Left.W*Right.Z + Left.X*Right.Y - Left.Y*Right.X + Left.Z*Right.W);
   end "*";
   
   function Rotate(V : Vector3; Q : Unit_Quaternion) return Vector3 is
      Pure_Quat : constant Quaternion := (W => 0.0, X => V.X, Y => V.Y, Z => V.Z);
      Conj      : constant Unit_Quaternion := Conjugate(Q);
      -- q * v * q^-1
      Tmp       : constant Quaternion := Q * Pure_Quat;
      Rotated   : constant Quaternion := Tmp * Conj;
   begin
      return (X => Rotated.X, Y => Rotated.Y, Z => Rotated.Z);
   end Rotate;
   
   function From_Axis_Angle(Axis : Vector3; Angle : Real) return Unit_Quaternion is
      Normalized_Axis : constant Vector3 := Normalize(Axis);
      Half_Angle      : constant Real := Angle / 2.0;
      Sin_Half        : constant Real := Math.Sin(Half_Angle);
   begin
      return 
        (W => Math.Cos(Half_Angle),
         X => Normalized_Axis.X * Sin_Half,
         Y => Normalized_Axis.Y * Sin_Half,
         Z => Normalized_Axis.Z * Sin_Half);
   end From_Axis_Angle;

end Quaternion_Rotation_Math;

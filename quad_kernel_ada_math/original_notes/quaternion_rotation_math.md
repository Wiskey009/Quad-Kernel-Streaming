# quaternion_rotation_math



```ada
--  quaternion_rotation_math.ads
with Ada.Numerics.Generic_Elementary_Functions;

package Quaternion_Rotation_Math is
   pragma Pure;
   pragma Assertion_Policy(Pre => Check, Post => Check, Type_Invariant => Check);
   
   type Real is digits 15;
   package Math is new Ada.Numerics.Generic_Elementary_Functions(Real);
   
   type Vector3 is record
      X, Y, Z : Real := 0.0;
   end record
     with Type_Invariant => Is_Finite(Vector3);
   
   type Quaternion is record
      W, X, Y, Z : Real := 0.0;
   end record
     with Dynamic_Predicate => Is_Finite(Quaternion),
          Type_Invariant => Is_Finite(Quaternion);
   
   subtype Unit_Quaternion is Quaternion
     with Dynamic_Predicate => Is_Unit(Unit_Quaternion);
   
   function Is_Finite(V : Vector3) return Boolean is
     (V.X'Valid and V.Y'Valid and V.Z'Valid);
   
   function Is_Finite(Q : Quaternion) return Boolean is
     (Q.W'Valid and Q.X'Valid and Q.Y'Valid and Q.Z'Valid);
   
   function Is_Unit(Q : Quaternion) return Boolean is
     (abs (Q.W**2 + Q.X**2 + Q.Y**2 + Q.Z**2 - 1.0) <= 1.0E-12)
       with Pre => Is_Finite(Q);
   
   function Identity return Unit_Quaternion is
     (W => 1.0, X => 0.0, Y => 0.0, Z => 0.0)
       with Post => Is_Unit(Identity'Result);
   
   function Conjugate(Q : Unit_Quaternion) return Unit_Quaternion is
     (W => Q.W, X => -Q.X, Y => -Q.Y, Z => -Q.Z)
       with Post => Is_Unit(Conjugate'Result);
   
   function Normalize(Q : Quaternion) return Unit_Quaternion
     with Pre => Norm(Q) > Real'Epsilon,
          Post => Is_Unit(Normalize'Result);
   
   function "*" (Left, Right : Unit_Quaternion) return Unit_Quaternion
     with Post => Is_Unit("*"'Result);
   
   function Rotate(V : Vector3; Q : Unit_Quaternion) return Vector3
     with Post => Distance(V, Rotate'Result) <= 2.0 * Real'Model_Epsilon * Norm(V);
   
   function From_Axis_Angle(Axis : Vector3; Angle : Real) return Unit_Quaternion
     with Pre => Norm(Axis) > Real'Epsilon and then Is_Unit(Normalize(Axis)),
          Post => Is_Unit(From_Axis_Angle'Result);
   
private
   function Norm(Q : Quaternion) return Real is
     (Math.Sqrt(Q.W**2 + Q.X**2 + Q.Y**2 + Q.Z**2))
       with Pre => Is_Finite(Q);
   
   function Norm(V : Vector3) return Real is
     (Math.Sqrt(V.X**2 + V.Y**2 + V.Z**2))
       with Pre => Is_Finite(V);
   
   function Normalize(V : Vector3) return Vector3 is
     (V.X / Norm(V), V.Y / Norm(V), V.Z / Norm(V))
       with Pre => Norm(V) > Real'Epsilon,
            Post => abs (Norm(Normalize'Result) - 1.0) <= 1.0E-12;
   
   function Distance(U, V : Vector3) return Real is
     (Norm((U.X - V.X, U.Y - V.Y, U.Z - V.Z)))
       with Pre => Is_Finite(U) and Is_Finite(V);
end Quaternion_Rotation_Math;

--  quaternion_rotation_math.adb
package body Quaternion_Rotation_Math is
   function Normalize(Q : Quaternion) return Unit_Quaternion is
      Magnitude : constant Real := Norm(Q);
   begin
      return (W => Q.W / Magnitude,
              X => Q.X / Magnitude,
              Y => Q.Y / Magnitude,
              Z => Q.Z / Magnitude);
   end Normalize;
   
   function "*" (Left, Right : Unit_Quaternion) return Unit_Quaternion is
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
      Rotated   : constant Quaternion := Q * Pure_Quat * Conj;
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

-- Validation Tests (partial)
-- test_quaternion_rotation.adb
with Quaternion_Rotation_Math; use Quaternion_Rotation_Math;
with Ada.Assertions; use Ada.Assertions;
with Ada.Text_IO; use Ada.Text_IO;

procedure Test_Quaternion_Rotation is
   function Approx_Equal(U, V : Vector3; Epsilon : Real := 1.0E-6) return Boolean is
     (abs (U.X - V.X) <= Epsilon and
      abs (U.Y - V.Y) <= Epsilon and
      abs (U.Z - V.Z) <= Epsilon);
   
   Identity_Q : constant Unit_Quaternion := Identity;
   V : constant Vector3 := (1.0, 2.0, 3.0);
begin
   -- Identity test
   Assert(Approx_Equal(Rotate(V, Identity_Q), V), "Identity rotation failed");
   
   -- 180-degree X-axis rotation
   declare
      Q : constant Unit_Quaternion := From_Axis_Angle((1.0, 0.0, 0.0), 3.1415926535);
      Expected : constant Vector3 := (1.0, -2.0, -3.0);
   begin
      Assert(Approx_Equal(Rotate(V, Q), Expected), "X-180 rotation failed");
   end;
   
   -- Composition test
   declare
      Q1 : constant Unit_Quaternion := From_Axis_Angle((0.0, 1.0, 0.0), 1.5708);
      Q2 : constant Unit_Quaternion := From_Axis_Angle((1.0, 0.0, 0.0), 1.5708);
      V_Rot : Vector3 := Rotate(Rotate(V, Q1), Q2);
      Q_Composed : constant Unit_Quaternion := Q1 * Q2;
      V_Composed : constant Vector3 := Rotate(V, Q_Composed);
   begin
      Assert(Approx_Equal(V_Rot, V_Composed), "Composition test failed");
   end;
end Test_Quaternion_Rotation;

-- Performance Notes:
-- 1. Quaternion operations: O(1) constant time
-- 2. No dynamic allocations, stack-only
-- 3. Normalize is most expensive (sqrt), cache results
-- 4. Inlining beneficial for small functions
-- 5. FPU-intensive, benefits from hardware sqrt/cos/sin
```
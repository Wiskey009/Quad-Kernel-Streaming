with Ada.Numerics.Generic_Elementary_Functions;

package Quaternion_Rotation_Math is
   pragma Pure;
   
   type Real is digits 15;
   package Math is new Ada.Numerics.Generic_Elementary_Functions(Real);
   
   type Vector3 is record
      X, Y, Z : Real := 0.0;
   end record;
   
   type Quaternion is record
      W, X, Y, Z : Real := 0.0;
   end record;
   
   subtype Unit_Quaternion is Quaternion;
   
   function Identity return Unit_Quaternion;
   
   function Conjugate(Q : Unit_Quaternion) return Unit_Quaternion;
   
   function Norm(Q : Quaternion) return Real;
   function Norm(V : Vector3) return Real;

   function Normalize(Q : Quaternion) return Unit_Quaternion;
   function Normalize(V : Vector3) return Vector3;

   function "*" (Left, Right : Quaternion) return Quaternion;
   
   function Rotate(V : Vector3; Q : Unit_Quaternion) return Vector3;
   
   function From_Axis_Angle(Axis : Vector3; Angle : Real) return Unit_Quaternion;

end Quaternion_Rotation_Math;

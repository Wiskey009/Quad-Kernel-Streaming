package body Bezier_Spline_Engine is

   function Horner_Bezier (P0, P1, P2, P3 : Real; T : Unit_Interval) return Real is
      Inv_T : constant Real := 1.0 - T;
   begin
      return (Inv_T**3 * P0 + 3.0*Inv_T**2*T * P1 + 3.0*Inv_T*T**2 * P2 + T**3 * P3);
   end Horner_Bezier;

   --  Bezier Segment Evaluation
   function Evaluate_Segment (S : Bezier_Segment; T : Unit_Interval) return Point is
      Result : Point;
   begin
      Result.X := Horner_Bezier(S(0).X, S(1).X, S(2).X, S(3).X, T);
      Result.Y := Horner_Bezier(S(0).Y, S(1).Y, S(2).Y, S(3).Y, T);
      return Result;
   end Evaluate_Segment;

   --  Full Spline Evaluation
   function Evaluate_Spline (Spline : Bezier_Spline; T : Real) return Point is
      Idx : Natural := Natural(Real'Floor(T));
      Local_T : Real := T - Real(Idx);
   begin
      if Idx >= Spline'Length then
         Idx := Spline'Length - 1;
         Local_T := 1.0;
      elsif Idx < 0 then
         Idx := 0;
         Local_T := 0.0;
      end if;
      
      -- Handle the case where T is exactly Spline'Length
      if T = Real(Spline'Length) then
          Idx := Spline'Length - 1;
          Local_T := 1.0;
      end if;

      return Evaluate_Segment(Spline(Idx + Spline'First), Local_T);
   end Evaluate_Spline;

end Bezier_Spline_Engine;

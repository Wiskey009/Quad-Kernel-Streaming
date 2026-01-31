package body Statistical_Analysis_Module is

   function Mean(X : Element_Array) return Element is
      Sum : Long_Float := 0.0;
   begin
      for I in X'Range loop
         Sum := Sum + Long_Float(X(I));
      end loop;
      return Element(Sum / Long_Float(X'Length));
   end Mean;

   function Variance(X : Element_Array) return Element is
      function Local_Mean is new Mean(Element, Element_Array);
      Mu : constant Element := Local_Mean(X);
      Sum_Sq : Long_Float := 0.0;
   begin
      for I in X'Range loop
         Sum_Sq := Sum_Sq + (Long_Float(X(I)) - Long_Float(Mu))**2;
      end loop;
      return Element(Sum_Sq / Long_Float(X'Length - 1));
   end Variance;

   function Entropy(Probabilities : Probability_Array) return Real is
      H : Real := 0.0;
   begin
      for I in Probabilities'Range loop
         if Probabilities(I) > 0.0 then
            H := H - Real(Probabilities(I)) * Math.Log(Real(Probabilities(I)));
         end if;
      end loop;
      return H;
   end Entropy;

end Statistical_Analysis_Module;

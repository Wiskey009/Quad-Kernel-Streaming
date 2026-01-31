with Ada.Numerics.Generic_Elementary_Functions;

package Statistical_Analysis_Module with SPARK_Mode is
   pragma Pure;
   
   type Probability is digits 15 range 0.0..1.0;
   type Real is digits 15;
   type Index_Type is new Long_Integer range 1 .. Long_Integer'Last;
   type Index is range 1..Index_Type'Last; -- Fixed to a general index
   
   package Math is new Ada.Numerics.Generic_Elementary_Functions(Real);

   type Probability_Array is array (Positive range <>) of Probability;

   --  Mean: μ = (1/n)Σx_i
   generic
      type Element is digits <>;
      type Element_Array is array(Positive range <>) of Element;
   function Mean(X : Element_Array) return Element
     with Pre => X'Length >= 1;

   --  Variance: σ² = (1/(n-1))Σ(x_i - μ)^2
   generic
      type Element is digits <>;
      type Element_Array is array(Positive range <>) of Element;
   function Variance(X : Element_Array) return Element
     with Pre => X'Length >= 2;

   --  Entropy: H = -Σp_i ln p_i
   function Entropy(Probabilities : Probability_Array) return Real
     with Pre => (Probabilities'Length > 0 and then (for all J in Probabilities'Range => Probabilities(J) >= 0.0));

end Statistical_Analysis_Module;

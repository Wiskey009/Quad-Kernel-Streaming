package Convolution_Correlation is
   pragma Assertion_Policy (Check);

   generic
      type Real is digits <>;
      type Index_Type is range <>;
      type Signal_1D is array (Index_Type range <>) of Real;
      type Signal_2D is array (Index_Type range <>, Index_Type range <>) of Real;
   package Generic_Convolution is

      -- 1D Convolution: f * kernel
      function Convolve_1D
        (f      : Signal_1D;
         kernel : Signal_1D)
         return Signal_1D
      with
        Pre  => f'Length > 0 and kernel'Length > 0;

      -- 2D Convolution: I * kernel
      type Padding_Mode is (Zero, Replicate, Symmetric);
      
      function Convolve_2D
        (I      : Signal_2D;
         kernel : Signal_2D;
         Pad    : Padding_Mode := Zero)
         return Signal_2D
      with
        Pre  =>
          I'Length(1) > 0 and I'Length(2) > 0 and
          kernel'Length(1) > 0 and kernel'Length(2) > 0;

      -- Correlation functions
      function Correlate_1D (f, kernel : Signal_1D) return Signal_1D;
      function Correlate_2D (I, kernel : Signal_2D; Pad : Padding_Mode := Zero) return Signal_2D;

   end Generic_Convolution;

end Convolution_Correlation;

package body Convolution_Correlation is

   package body Generic_Convolution is

      function Convolve_1D
        (f      : Signal_1D;
         kernel : Signal_1D)
         return Signal_1D
      is
         Result_Len : constant Integer := f'Length + kernel'Length - 1;
         Result_Low : constant Index_Type := f'First;
         Result_High : constant Index_Type := Index_Type(Integer(f'First) + Result_Len - 1);
         Result : Signal_1D (Result_Low .. Result_High) := (others => 0.0);
         
         N : constant Integer := f'Length;
         M : constant Integer := kernel'Length;
      begin
         for i in 0 .. Result_Len - 1 loop
            declare
               Acc : Real := 0.0;
               n : constant Integer := i; -- index in result
            begin
               for K_Idx in 0 .. M - 1 loop
                  declare
                     -- Convolution formula: (f * g)[n] = Σ f[m] * g[n - m]
                     -- Or here: Σ f[n - m] * kernel[m]
                     idx_f_int : constant Integer := i - K_Idx;
                  begin
                     if idx_f_int >= 0 and idx_f_int < N then
                        declare
                           idx_f : constant Index_Type := Index_Type(Integer(f'First) + idx_f_int);
                           idx_k : constant Index_Type := Index_Type(Integer(kernel'First) + K_Idx);
                        begin
                           Acc := Acc + f(idx_f) * kernel(idx_k);
                        end;
                     end if;
                  end;
               end loop;
               Result(Index_Type(Integer(Result_Low) + i)) := Acc;
            end;
         end loop;
         return Result;
      end Convolve_1D;

      function Convolve_2D
        (I      : Signal_2D;
         kernel : Signal_2D;
         Pad    : Padding_Mode := Zero)
         return Signal_2D
      is
         Result : Signal_2D (I'Range(1), I'Range(2)) := (others => (others => 0.0));
         K1 : constant Integer := kernel'Length(1);
         K2 : constant Integer := kernel'Length(2);
         H1 : constant Integer := K1 / 2;
         H2 : constant Integer := K2 / 2;
      begin
         for r in I'Range(1) loop
            for c in I'Range(2) loop
               declare
                  Acc : Real := 0.0;
               begin
                  for kr in 0 .. K1 - 1 loop
                     for kc in 0 .. K2 - 1 loop
                        declare
                           ir : constant Integer := Integer(r) - (kr - H1);
                           ic : constant Integer := Integer(c) - (kc - H2);
                        begin
                           if ir in Integer(I'First(1)) .. Integer(I'Last(1)) and then
                              ic in Integer(I'First(2)) .. Integer(I'Last(2))
                           then
                              Acc := Acc + I(Index_Type(ir), Index_Type(ic)) * 
                                     kernel(Index_Type(Integer(kernel'First(1)) + kr),
                                            Index_Type(Integer(kernel'First(2)) + kc));
                           end if;
                        end;
                     end loop;
                  end loop;
                  Result(r, c) := Acc;
               end;
            end loop;
         end loop;
         return Result;
      end Convolve_2D;

      function Correlate_1D (f, kernel : Signal_1D) return Signal_1D is
         -- Correlation is convolution with flipped kernel
         Flipped_Kernel : Signal_1D(kernel'Range);
      begin
         for i in 0 .. kernel'Length - 1 loop
            Flipped_Kernel(Index_Type(Integer(kernel'First) + i)) := 
              kernel(Index_Type(Integer(kernel'Last) - i));
         end loop;
         return Convolve_1D(f, Flipped_Kernel);
      end Correlate_1D;

      function Correlate_2D (I, kernel : Signal_2D; Pad : Padding_Mode := Zero) return Signal_2D is
         Flipped_Kernel : Signal_2D(kernel'Range(1), kernel'Range(2));
         K1 : constant Integer := kernel'Length(1);
         K2 : constant Integer := kernel'Length(2);
      begin
         for r in 0 .. K1 - 1 loop
            for c in 0 .. K2 - 1 loop
               Flipped_Kernel(Index_Type(Integer(kernel'First(1)) + r),
                              Index_Type(Integer(kernel'First(2)) + c)) :=
                 kernel(Index_Type(Integer(kernel'Last(1)) - r),
                        Index_Type(Integer(kernel'Last(2)) - c));
            end loop;
         end loop;
         return Convolve_2D(I, Flipped_Kernel, Pad);
      end Correlate_2D;

   end Generic_Convolution;

end Convolution_Correlation;

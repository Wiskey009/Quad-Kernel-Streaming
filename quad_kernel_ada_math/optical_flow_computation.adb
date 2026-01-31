package body Optical_Flow_Computation is
   
   -- Safe pixel access with boundary checking
   function Safe_Value (
      Img : Image_Matrix;
      i, j : Integer
   ) return Integer is
   begin
      if i in Img'Range(1) and j in Img'Range(2) then
         return Integer(Img(i, j));
      else
         return 0;
      end if;
   end Safe_Value;

   -- Lucas-Kanade Optical Flow Implementation
   procedure Compute_Flow (
      Previous, Current : in Image_Matrix;
      Flow              : out Flow_Field;
      Window_Size       : Natural := 5
   ) is
      Half_Window : constant Natural := Window_Size / 2;
      
      -- Use Long_Float for internal calculations to avoid fixed-point ambiguity
      subtype Accum is Long_Float;
   begin
      for x in Previous'Range(1) loop
         for y in Previous'Range(2) loop
            declare
               -- Structure tensor components (A^T * A)
               Sum_Ix2  : Accum := 0.0;
               Sum_Iy2  : Accum := 0.0;
               Sum_IxIy : Accum := 0.0;
               -- Right-hand side (A^T * b)
               Sum_IxIt : Accum := 0.0;
               Sum_IyIt : Accum := 0.0;
            begin
               -- Accumulate over the window
               for wi in -Integer(Half_Window) .. Integer(Half_Window) loop
                  for wj in -Integer(Half_Window) .. Integer(Half_Window) loop
                     declare
                        -- Spatial gradients using central differences
                        Ix_Int : constant Integer := 
                          Safe_Value(Previous, x + wi + 1, y + wj) - 
                          Safe_Value(Previous, x + wi - 1, y + wj);
                        
                        Iy_Int : constant Integer := 
                          Safe_Value(Previous, x + wi, y + wj + 1) - 
                          Safe_Value(Previous, x + wi, y + wj - 1);
                        
                        -- Temporal gradient
                        It_Int : constant Integer := 
                          Safe_Value(Current, x + wi, y + wj) - 
                          Safe_Value(Previous, x + wi, y + wj);
                           
                        -- Convert to floating point, scale derivatives
                        Ix : constant Accum := Accum(Ix_Int) / 2.0;
                        Iy : constant Accum := Accum(Iy_Int) / 2.0;
                        It : constant Accum := Accum(It_Int);
                     begin
                        -- Accumulate structure tensor
                        Sum_Ix2  := Sum_Ix2  + Ix * Ix;
                        Sum_Iy2  := Sum_Iy2  + Iy * Iy;
                        Sum_IxIy := Sum_IxIy + Ix * Iy;
                        Sum_IxIt := Sum_IxIt + Ix * It;
                        Sum_IyIt := Sum_IyIt + Iy * It;
                     end;
                  end loop;
               end loop;

               -- Solve the 2x2 linear system:
               -- [Sum_Ix2  Sum_IxIy] [u]   [-Sum_IxIt]
               -- [Sum_IxIy Sum_Iy2 ] [v] = [-Sum_IyIt]
               declare
                  Det : constant Accum := Sum_Ix2 * Sum_Iy2 - Sum_IxIy * Sum_IxIy;
                  u : Flow_Component;
                  v : Flow_Component;
               begin
                  if abs(Det) > 1.0e-6 then
                     -- Cramer's rule for 2x2 system
                     declare
                        Inv_Det : constant Accum := 1.0 / Det;
                        U_Val : constant Accum := 
                          (-Sum_Iy2 * Sum_IxIt + Sum_IxIy * Sum_IyIt) * Inv_Det;
                        V_Val : constant Accum := 
                          (Sum_IxIy * Sum_IxIt - Sum_Ix2 * Sum_IyIt) * Inv_Det;
                     begin
                        u := Flow_Component(U_Val);
                        v := Flow_Component(V_Val);
                     end;
                  else
                     -- Singular matrix: aperture problem, zero flow
                     u := 0.0;
                     v := 0.0;
                  end if;

                  Flow(x, y) := (u, v);
               end;
            end;
         end loop;
      end loop;
   end Compute_Flow;
   
end Optical_Flow_Computation;

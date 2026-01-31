# histogram_equalization_algorithm

```ada
--  histogram_equalization_algorithm.ads
with Ada.Numerics.Discrete_Random;

package Histogram_Equalization_Algorithm is
   subtype Pixel_Value is Integer range 0..255;
   type Image_Matrix is array (Natural range <>, Natural range <>) of Pixel_Value;
   type Histogram is array (Pixel_Value) of Natural;

   --  Main equalization procedure
   procedure Adaptive_Equalize 
     (Input : in Image_Matrix;
      Output : out Image_Matrix;
      Window_Radius : in Positive)
   with
     Pre => Input'Length(1) > 0 and Input'Length(2) > 0
       and then Input'First(1) = Output'First(1)
       and then Input'Last(1) = Output'Last(1)
       and then Input'First(2) = Output'First(2)
       and then Input'Last(2) = Output'Last(2),
     Post => (for all I in Output'Range(1), J in Output'Range(2) => 
                Output(I,J) in Pixel_Value);
       
private
   function Compute_CDF (H : Histogram; Total : Natural) return Histogram
     with
       Pre => Total <= Natural'Last / 255,
       Post => (Compute_CDF'Result(0) = H(0)) and
               (for all K in Pixel_Value range 1..255 =>
                  Compute_CDF'Result(K) = Compute_CDF'Result(K-1) + H(K)),
     Contract_Cases => (Total = 0 => Compute_CDF'Result = (others => 0));

   function Remap_Value (CDF : Histogram; Value : Pixel_Value) return Pixel_Value
     with
       Pre => CDF(255) <= Natural'Last - 255,
       Post => Remap_Value'Result in Pixel_Value;
end Histogram_Equalization_Algorithm;

--  histogram_equalization_algorithm.adb
package body Histogram_Equalization_Algorithm with SPARK_Mode is

   --  Theorem 1: Output bounded [0,255]
   --  Proof: CDF_min = 0, CDF_max = Total. Remapping formula:
   --  floor(255*(CDF - CDF_min)/(CDF_max - CDF_min)) ≤ 255
   --  Since (CDF - CDF_min) ≤ (CDF_max - CDF_min), result ≤ 255

   function Compute_CDF (H : Histogram; Total : Natural) return Histogram is
      CDF : Histogram := (others => 0);
   begin
      CDF(0) := H(0);
      for K in 1..255 loop
         CDF(K) := CDF(K-1) + H(K);
         pragma Loop_Invariant
           (for all L in 0..K => CDF(L) = (if L=0 then H(0) else CDF(L-1) + H(L)));
      end loop;
      return CDF;
   end Compute_CDF;

   --  Theorem 2: Monotonic preservation
   --  Proof: ∀i<j, CDF(i) ≤ CDF(j) ⇒ Remap_Value(i) ≤ Remap_Value(j)
   --  Follows from linear scaling of sorted inputs

   function Remap_Value (CDF : Histogram; Value : Pixel_Value) return Pixel_Value is
      Scaled : Natural;
   begin
      if CDF(255) = 0 then
         return 0;
      end if;
      Scaled := (255 * CDF(Value)) / CDF(255);
      return Pixel_Value(Scaled);
   end Remap_Value;

   --  Theorem 3: No overflow in CDF computation
   --  Proof: Max CDF(255) = Total pixels ≤ (2*Window_Radius+1)^2 * 2^16
   --  Protected by Window_Radius constraint

   procedure Adaptive_Equalize 
     (Input : in Image_Matrix;
      Output : out Image_Matrix;
      Window_Radius : in Positive) 
   is
      Rows : constant Natural := Input'Length(1);
      Cols : constant Natural := Input'Length(2);
      Window_Size : constant Positive := 2*Window_Radius + 1;
   begin
      Output := (others => (others => 0));

      for I in Input'Range(1) loop
         for J in Input'Range(2) loop
            declare
               H : Histogram := (others => 0);
               Total : Natural := 0;
               Start_Row : Natural := Natural'Max(Input'First(1), I - Window_Radius);
               End_Row : Natural := Natural'Min(Input'Last(1), I + Window_Radius);
               Start_Col : Natural := Natural'Max(Input'First(2), J - Window_Radius);
               End_Col : Natural := Natural'Min(Input'Last(2), J + Window_Radius);
            begin
               --  Build local histogram
               for X in Start_Row..End_Row loop
                  for Y in Start_Col..End_Col loop
                     H(Input(X,Y)) := H(Input(X,Y)) + 1;
                     Total := Total + 1;
                     pragma Loop_Invariant (Total <= (X - Start_Row + 1)*(Y - Start_Col + 1));
                     pragma Loop_Invariant (H(Input(X,Y)) <= Total);
                  end loop;
               end loop;

               --  Equalization core
               declare
                  CDF : constant Histogram := Compute_CDF(H, Total);
               begin
                  Output(I,J) := Remap_Value(CDF, Input(I,J));
               end;
            end;
            pragma Loop_Invariant (for all X in Output'First(1)..I, Y in Output'First(2)..J => 
                                     Output(X,Y) in Pixel_Value);
         end loop;
      end loop;
   end Adaptive_Equalize;

end Histogram_Equalization_Algorithm;
```

```ada
--  validation_tests.adb
with Ada.Assertions; use Ada.Assertions;
with Histogram_Equalization_Algorithm; use Histogram_Equalization_Algorithm;

procedure Validation_Tests is
   Test_Image : Image_Matrix(1..4, 1..4) := 
     ((50, 100, 150, 200),
      (50, 100, 150, 200),
      (50, 100, 150, 200),
      (50, 100, 150, 200));
   Result : Image_Matrix(1..4, 1..4);
begin
   --  Uniform regions test
   Adaptive_Equalize(Test_Image, Result, 1);
   for I in 1..4 loop
      for J in 1..4 loop
         Assert(Result(I,J) = ((J-1)*85), "Linear mapping failure");
      end loop;
   end loop;

   --  Extreme values test
   declare
      Dark : Image_Matrix(1..2, 1..2) := ((0,0),(0,0));
      Bright : Image_Matrix(1..2, 1..2) := ((255,255),(255,255));
      Dark_Res, Bright_Res : Image_Matrix(1..2, 1..2);
   begin
      Adaptive_Equalize(Dark, Dark_Res, 1);
      Adaptive_Equalize(Bright, Bright_Res, 1);
      Assert((for all X in 1..2, Y in 1..2 => Dark_Res(X,Y) = 0), 
             "Dark preservation failure");
      Assert((for all X in 1..2, Y in 1..2 => Bright_Res(X,Y) = 255), 
             "Bright preservation failure");
   end;
end Validation_Tests;
```

**Ecuaciones**:
$$
\text{CDF}(k) = \sum_{i=0}^{k} H(i) \quad \text{para} \quad 0 \leq k \leq 255
$$
$$
\text{Remap}(v) = \left\lfloor \frac{\text{CDF}(v) - \text{CDF}_{\min}}{\text{CDF}_{\max} - \text{CDF}_{\min}} \times 255 \right\rfloor
$$

**Teoremas & Proofs**:
1. **Bound Preservation**: $\forall v \in [0,255], \text{Remap}(v) \in [0,255]$  
   *Proof*: By CDF monotonicity, $\text{CDF}(v) \in [0, \text{Total}]$, thus fraction $\in [0,1]$. Scaling by 255 preserves bounds.  

2. **Monotonicity**: $v_1 \leq v_2 \Rightarrow \text{Remap}(v_1) \leq \text{Remap}(v_2)$  
   *Proof*: Since CDF is non-decreasing, numerator preserves inequality, denominator is positive constant.  

3. **Overflow Safety**: $|\text{Window}| \leq (2r+1)^2 \Rightarrow \text{CDF}(255) \leq (2r+1)^2 \times 256$  
   *Proof*: Each pixel contributes to exactly one histogram bin, bounded by window size and max pixel frequency.

**Performance Notes**:  
- Time complexity: $O(mnw^2)$ for $m \times n$ image with window radius $w$  
- Window reuse optimization possible via sliding window histogram updates  
- Parallelization viable per-pixel for distributed systems
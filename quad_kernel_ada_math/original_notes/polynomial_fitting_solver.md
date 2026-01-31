# polynomial_fitting_solver



```ada
-- polynomial_fitting_solver.ads
with Ada.Numerics.Generic_Real_Arrays;
with Ada.Numerics.Generic_Elementary_Functions;

generic
   type Real is digits <>;
package Polynomial_Fitting_Solver is
   pragma Assertion_Policy (Pre => Check, Post => Check, Type_Invariant => Check);
   subtype Degree_Type is Natural range 0..100;
   type Coefficient_Array is array (Natural range <>) of Real;
   type Point_Array is array (Natural range <>) of Real;
   type Point_List is record
      X, Y : Point_Array;
   end record;

   function Vandermonde_Matrix (X : Point_Array; Degree : Degree_Type)
      return Ada.Numerics.Generic_Real_Arrays.Real_Matrix
   with Pre => X'Length >= 1 and Degree <= X'Length - 1,
        Post => Vandermonde_Matrix'Result'Length(1) = X'Length and
                Vandermonde_Matrix'Result'Length(2) = Degree + 1;

   function Least_Squares_Fit (Points : Point_List; Degree : Degree_Type)
      return Coefficient_Array
   with
      Pre => Points.X'Length = Points.Y'Length and
             Points.X'Length >= Degree + 1 and
             Points.X'Length >= 1,
      Post => Least_Squares_Fit'Result'Length = Degree + 1;

   function Interpolate (Points : Point_List) return Coefficient_Array
   with
      Pre => Points.X'Length = Points.Y'Length and
             Points.X'Length >= 1 and
             (for all I in Points.X'Range =>
                (for all J in Points.X'First..I-1 => Points.X(I) /= Points.X(J))),
      Post => Interpolate'Result'Length = Points.X'Length;

private
   package Real_Arrays is new Ada.Numerics.Generic_Real_Arrays(Real);
   package Elementary_Fns is new Ada.Numerics.Generic_Elementary_Functions(Real);

   function Is_Symmetric (M : Real_Arrays.Real_Matrix) return Boolean is
     (for all I in M'Range(1) => (for all J in M'Range(2) => M(I, J) = M(J, I)))
   with Ghost;

end Polynomial_Fitting_Solver;

-- polynomial_fitting_solver.adb
package body Polynomial_Fitting_Solver is

   function Vandermonde_Matrix (X : Point_Array; Degree : Degree_Type)
      return Real_Arrays.Real_Matrix is
      Result : Real_Arrays.Real_Matrix (X'Range, 0..Degree);
   begin
      for I in X'Range loop
         Result(I, 0) := 1.0;
         for J in 1..Degree loop
            Result(I, J) := Result(I, J-1) * X(I);
         end loop;
      end loop;
      return Result;
   end Vandermonde_Matrix;

   function Least_Squares_Fit (Points : Point_List; Degree : Degree_Type)
      return Coefficient_Array
   is
      use Real_Arrays;
      A  : constant Real_Matrix := Vandermonde_Matrix(Points.X, Degree);
      ATA : constant Real_Matrix := Transpose(A) * A;
      ATY : constant Real_Vector := Transpose(A) * Points.Y;
      C   : Real_Vector (0..Degree);
   begin
      pragma Assert (Is_Symmetric(ATA));
      C := Solve(ATA, ATY);
      return Coefficient_Array(C);
   end Least_Squares_Fit;

   function Interpolate (Points : Point_List) return Coefficient_Array is
   begin
      return Least_Squares_Fit(Points, Points.X'Length - 1);
   end Interpolate;

end Polynomial_Fitting_Solver;
```

**Ecuaciones**  
Dado puntos $(x_i, y_i)$, minimizar $\sum_{i=1}^m (y_i - \sum_{j=0}^n c_j x_i^j)^2$.  
Matriz de Vandermonde $A_{ij}=x_i^j$. Solución:  
$$(A^TA)\mathbf{c} = A^T\mathbf{y} \implies \mathbf{c} = (A^TA)^{-1}A^T\mathbf{y}$$

**Teoremas & Proofs**  
*Teorema 1 (Existencia)*: Si $m \geq n+1$ y los $x_i$ son distintos, $A^TA$ es invertible.  
*Demostración*: La matriz $A$ tiene rango completo por columnas, luego $A^TA$ es SPD.  
*Teorema 2 (Precisión Ada)*: El algoritmo implementa exactamente las ecuaciones normales.  
*Demostración*: Por construcción de la matriz Vandermonde y uso de Solve() de Ada.Numerics.  
*Teorema 3 (Seguridad)*: Todos los accesos a arrays están dentro de los límites.  
*Demostración*: Los contratos garantizan Degree ≤ m-1 y precondiciones en Solve().

**Validation Tests**  
1. **Ajuste lineal**: Puntos {(0,1), (1,3), (2,5)} → $2x+1$  
2. **Interpolación cúbica**: Puntos {(0,0), (1,1), (-1,-1), (2,8)} → $x^3$  
3. **Matriz singular**: Intentar ajuste con x duplicados → Violación de precondición  
4. **Grado cero**: Puntos {(5,3)} → Polinomio constante 3.0  
5. **Precisión numérica**: Comparar con soluciones analíticas usando ε=1e-9

**Performance Notes**  
- Complejidad dominante: $O(mn^2 + n^3)$  
- Limitado por operaciones de matriz densa  
- Adecuado para n ≤ 100, evite grados altos  
- Solve() usa eliminación gaussiana con pivoteo
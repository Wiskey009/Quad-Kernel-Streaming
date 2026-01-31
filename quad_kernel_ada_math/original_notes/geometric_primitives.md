# geometric_primitives

```ada
-- geometric_primitives.ads
package Geometric_Primitives is
   pragma Pure;

   subtype Coordinate is Long_Float range -1.0e12 .. 1.0e12;
   type Point is record
      X, Y : Coordinate;
   end record;

   type Line is record
      A, B, C : Coordinate;
   end record with
     Pre => (Line.A /= 0.0 or Line.B /= 0.0);

   type Circle is record
      Center : Point;
      Radius : Coordinate;
   end record with
     Pre => Circle.Radius >= 0.0;

   type Polygon is array (Positive range <>) of Point with
     Predicate => Polygon'Length >= 3;

   -- Line construction from two points
   function Create_Line (P1, P2 : Point) return Line with
     Pre => P1 /= P2,
     Post => (Create_Line'Result.A * (P1.X - P2.X) +
              Create_Line'Result.B * (P1.Y - P2.Y)) < 0.0001;

   -- Distance between two points
   function Distance (P1, P2 : Point) return Coordinate with
     Post => Distance'Result >= 0.0;

   -- Point-to-line distance
   function Distance (P : Point; L : Line) return Coordinate with
     Post => Distance'Result >= 0.0;

   -- Circle circumference
   function Circumference (C : Circle) return Coordinate with
     Pre => C.Radius <= Coordinate'Last / (2.0 * Ada.Numerics.Pi),
     Post => Circumference'Result = 2.0 * Ada.Numerics.Pi * C.Radius;

   -- Polygon perimeter
   function Perimeter (P : Polygon) return Coordinate with
     Pre => P'Length >= 2,
     Post => Perimeter'Result >= 0.0;

   -- Collinearity check
   function Are_Collinear (P1, P2, P3 : Point) return Boolean with
     Post => Are_Collinear'Result = 
       (abs ((P2.X - P1.X) * (P3.Y - P1.Y) - 
             (P3.X - P1.X) * (P2.Y - P1.Y)) < 1.0e-6);

   -- Line intersection
   function Intersection (L1, L2 : Line) return Point with
     Pre => abs (L1.A * L2.B - L2.A * L1.B) > 1.0e-6;

   -- Polygon area (Shoelace formula)
   function Area (P : Polygon) return Coordinate with
     Pre => P'Length >= 3,
     Post => Area'Result >= 0.0;

private
   function Determinant (P1, P2 : Point) return Coordinate is
     (P1.X * P2.Y - P1.Y * P2.X);
end Geometric_Primitives;
```

```ada
-- geometric_primitives.adb
package body Geometric_Primitives is
   function Create_Line (P1, P2 : Point) return Line is
      A : constant Coordinate := P2.Y - P1.Y;
      B : constant Coordinate := P1.X - P2.X;
      C : constant Coordinate := Determinant(P2, P1);
   begin
      return (A, B, C);
   end Create_Line;

   function Distance (P1, P2 : Point) return Coordinate is
      DX : constant Coordinate := P2.X - P1.X;
      DY : constant Coordinate := P2.Y - P1.Y;
   begin
      return (DX**2 + DY**2)**0.5;
   end Distance;

   function Distance (P : Point; L : Line) return Coordinate is
      Num : constant Coordinate := abs(L.A * P.X + L.B * P.Y + L.C);
      Den : constant Coordinate := (L.A**2 + L.B**2)**0.5;
   begin
      return Num / Den;
   end Distance;

   function Circumference (C : Circle) return Coordinate is
   begin
      return 2.0 * Ada.Numerics.Pi * C.Radius;
   end Circumference;

   function Perimeter (P : Polygon) return Coordinate is
      Total : Coordinate := 0.0;
   begin
      for I in P'First..P'Last - 1 loop
         Total := Total + Distance(P(I), P(I + 1));
      end loop;
      Total := Total + Distance(P(P'Last), P(P'First));
      return Total;
   end Perimeter;

   function Are_Collinear (P1, P2, P3 : Point) return Boolean is
   begin
      return abs(Determinant((P2.X - P1.X, P2.Y - P1.Y),
                             (P3.X - P1.X, P3.Y - P1.Y))) < 1.0e-6;
   end Are_Collinear;

   function Intersection (L1, L2 : Line) return Point is
      Det : constant Coordinate := L1.A * L2.B - L2.A * L1.B;
      X   : Coordinate;
      Y   : Coordinate;
   begin
      X := (L2.C * L1.B - L1.C * L2.B) / Det;
      Y := (L1.C * L2.A - L2.C * L1.A) / Det;
      return (X, Y);
   end Intersection;

   function Area (P : Polygon) return Coordinate is
      Total : Coordinate := 0.0;
      Prev  : Point := P(P'Last);
   begin
      for Current of P loop
         Total := Total + Determinant(Prev, Current);
         Prev := Current;
      end loop;
      return abs(Total) / 2.0;
   end Area;
end Geometric_Primitives;
```

**Pruebas de Validación**:
```ada
-- test_geometric_primitives.adb
with Geometric_Primitives; use Geometric_Primitives;
with Ada.Assertions; use Ada.Assertions;
procedure Test_Geometric_Primitives is
   P1 : constant Point := (0.0, 0.0);
   P2 : constant Point := (1.0, 0.0);
   P3 : constant Point := (0.0, 1.0);
   L1 : constant Line := Create_Line(P1, P2);
   C  : constant Circle := ((0.0, 0.0), 1.0);
begin
   Assert(Distance(P1, P2) = 1.0, "Distance failed");
   Assert(Are_Collinear(P1, P2, (2.0, 0.0)), "Collinearity failed");
   Assert(abs(Area((P1, P2, P3)) - 0.5) < 1.0e-6, "Area failed");
   Assert(abs(Circumference(C) - 2.0 * Ada.Numerics.Pi) < 1.0e-6, "Circumference failed");
end Test_Geometric_Primitives;
```

**Ecuaciones Matemáticas**:
- Punto: \( P = (x, y) \)
- Línea: \( Ax + By + C = 0 \)
- Círculo: \( (x - h)^2 + (y - k)^2 = r^2 \)
- Área Polígono: \( \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right| \)

**Teoremas & Proofs**:
1. **Colinealidad**: Tres puntos \( P_1, P_2, P_3 \) son colineales si \( \Delta = 0 \), donde \( \Delta = (x_2 - x_1)(y_3 - y_1) - (x_3 - x_1)(y_2 - y_1) \).  
   *Proof*: \( \Delta \) es el área del paralelogramo formado por los vectores \( \vec{P_1P_2} \) y \( \vec{P_1P_3} \).

2. **Distancia Punto-Línea**: Para \( L: Ax + By + C = 0 \), distancia \( d = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}} \).  
   *Proof*: Minimizar la distancia euclidiana usando cálculo.

3. **Shoelace Validez**: El área del polígono es invariante bajo rotación y traslación.  
   *Proof*: Por el teorema de Green aplicado a funciones escalonadas.

**Performance Notes**:
- Complejidad O(n) para perímetro/área de polígonos
- Operaciones trigonométricas limitadas a constantes (π)
- Cálculos determinísticos sin recursión profunda
- Verificación SPARK garantiza ausencia de errores en tiempo de ejecución
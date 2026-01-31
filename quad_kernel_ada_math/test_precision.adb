with Ada.Text_IO; use Ada.Text_IO;
with Precision_Arithmetic_Lib; use Precision_Arithmetic_Lib;

procedure Test_Precision is
begin
   if Almost_Equal_Fixed(1.0, 1.0001, 0.001) then
      Put_Line("Fixed match passed");
   else
      Put_Line("Fixed match failed");
   end if;

   if Almost_Equal_Float(1.0, 1.0000001, 1.0e-6, 1.0e-8) then
      Put_Line("Float match passed");
   else
      Put_Line("Float match failed");
   end if;

   Put_Line("Precision arithmetic tests finished.");
end Test_Precision;

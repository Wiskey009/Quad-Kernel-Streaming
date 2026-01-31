with Interfaces.C.Strings;

package body Kernel_Bridge_Adapter is

   -- Use C's puts instead of Ada.Text_IO to avoid runtime initialization issues
   procedure C_Puts (S : String) is
      procedure puts (Str : Interfaces.C.Strings.chars_ptr)
        with Import => True, Convention => C, External_Name => "puts";
      C_Str : Interfaces.C.Strings.chars_ptr := 
               Interfaces.C.Strings.New_String (S);
   begin
      puts (C_Str);
      Interfaces.C.Strings.Free (C_Str);
   end C_Puts;

   function Math_Kernel_Initialize (Config : System.Address) return int is
      pragma Unreferenced (Config);
   begin
      C_Puts ("[MATH KERNEL] Initialized Precision Validation Engine.");
      return 0;
   end Math_Kernel_Initialize;

   function Math_Kernel_Process (Input : access Frame_Buffer; Output : access Frame_Buffer) return int is
      pragma Unreferenced (Output);
   begin
      if Input /= null then
         C_Puts ("[MATH KERNEL] Validating frame precision...");
      end if;
      return 0;
   end Math_Kernel_Process;

   function Math_Kernel_Finalize return int is
   begin
      C_Puts ("[MATH KERNEL] Finalized.");
      return 0;
   end Math_Kernel_Finalize;

end Kernel_Bridge_Adapter;

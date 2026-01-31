with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with System;

package Kernel_Bridge_Adapter with SPARK_Mode => On is

   type Frame_Buffer is record
      Data         : System.Address;
      Size         : size_t;
      Width        : int;
      Height       : int;
      Channels     : int;
      Sample_Rate  : int;
      Timestamp    : unsigned_long_long;
      Flags        : unsigned;
   end record;
   pragma Convention (C, Frame_Buffer);

   function Math_Kernel_Initialize (Config : System.Address) return int
     with Export => True, Convention => C, External_Name => "math_kernel_initialize";

   function Math_Kernel_Process (Input : access Frame_Buffer; Output : access Frame_Buffer) return int
     with Pre => Input /= null and Output /= null,
          Export => True, Convention => C, External_Name => "math_kernel_process";

   function Math_Kernel_Finalize return int
     with Export => True, Convention => C, External_Name => "math_kernel_finalize";

   type Kernel_Interface is record
      Initialize : System.Address;
      Process    : System.Address;
      Finalize   : System.Address;
   end record;
   pragma Convention (C, Kernel_Interface);

end Kernel_Bridge_Adapter;

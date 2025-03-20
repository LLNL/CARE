.. ##############################################################################
   # Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
   # project contributors. See the CARE LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ##############################################################################

===========
Loop Fusion
===========

Some applications launch so many small kernels that kernel launch overhead becomes quite a problem. CARE effectively addresses this issue with loop fusion. Some sample code is provided below to show how loop fusion can be introduced with minimal changes to the application source code.

.. code-block:: c++

   #include "care/care.h"
   #include "care/LoopFuser.h"

   int main(int argc, char* argv[]) {
      // Set up
      int numLoops = 1000000;
      int loopLength = 32;

      int_ptr a(loopLength,"a");
      int_ptr b(loopLength,"b");

      // This indicates the start of a fusible region. If this is not present,
      // FUSIBLE_LOOP_STREAM falls back to a normal STREAM loop.
      FUSIBLE_LOOPS_START

      for (int i = 0; i < numLoops; ++i) {
         // This loop will stored and saved for when the fusible region is ended
         FUSIBLE_LOOP_STREAM(j, 0, loopLength) {
            a[j] = i;
            b[j] = i/2;
         } FUSIBLE_LOOP_STREAM_END
      }

      // This indicates the end of a fusible region and launches a single kernel
      FUSIBLE_LOOPS_STOP

      // Clean up
      a.free();
      b.free();
      care::syncIfNeeded();

      return 0;
   }

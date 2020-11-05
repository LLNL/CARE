//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////


#include "care/config.h"

#if CARE_ENABLE_LOOP_FUSER

#define GPU_ACTIVE
#define HAVE_FUSER_TEST CARE_ENABLE_LOOP_FUSER
#if HAVE_FUSER_TEST
// always have DEBUG on to force the packer to be on for CPU builds.
#ifdef CARE_DEBUG
#undef CARE_DEBUG
#endif
#define CARE_DEBUG 1
#include "gtest/gtest.h"

// define if we want to test running loops as we encounter them
//#define CARE_FUSIBLE_LOOPS_DISABLE
// define if we want to turn on verbosity
//#define FUSER_VERBOSE

#include "care/LoopFuser.h"


// This makes it so we can use device lambdas from within a GPU_TEST
#define GPU_TEST(X, Y) static void gpu_test_ ## X_ ## Y(); \
   TEST(X, Y) { gpu_test_ ## X_ ## Y(); } \
   static void gpu_test_ ## X_ ## Y()

TEST(UpperBound_binarySearch, checkOffsets) {
   // These come from a segfault that occurred during development
   int offsetArr[] = { 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 272, 289, 306, 323, 340 };
   int offset;

   offset = care::binarySearch<int>(offsetArr, 0, 1, 0, true);
   EXPECT_EQ(offsetArr[offset], 17);

   offset = care::binarySearch<int>(offsetArr, 0, 20, 37, true);
   EXPECT_EQ(offsetArr[offset], 51);

   offset = care::binarySearch<int>(offsetArr, 0, 20, 33, true);
   EXPECT_EQ(offsetArr[offset], 34);

   offset = care::binarySearch<int>(offsetArr, 0, 20, 34, true);
   EXPECT_EQ(offsetArr[offset], 51);

   offset = care::binarySearch<int>(offsetArr, 0, 20, 339, true);
   EXPECT_EQ(offsetArr[offset], 340);

   offset = care::binarySearch<int>(offsetArr, 0, 20, 340, true);
   EXPECT_EQ(offset, -1);
}

GPU_TEST(TestPacker, packFixedRange) {
   LoopFuser * packer = LoopFuser::getInstance();
   packer->startRecording();
  
   int arrSize = 1024;
   care::host_device_ptr<int> src(arrSize);
   care::host_device_ptr<int> dst(arrSize);

   // initialize the src and dst on the host
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      if (i == 0) { 
         printf("seq: dst %p src %p\n", dst.data(chai::CPU), src.data(chai::CPU));
      }
      src[i] = i;
      dst[i] = -1;
   } CARE_SEQUENTIAL_LOOP_END

   int pos = 0;
   packer->registerAction(0, arrSize, pos,
                          [=] CARE_DEVICE(int, int *, int const*, int, fusible_registers) { },
                          [=] CARE_DEVICE(int i, int *, fusible_registers) {
      if (i == 0) {
         printf("dst %p src %p\n", dst.data(), src.data());
      }
      dst[pos+i] = src[i];
   });

   // pack has not been flushed, so
   // host data should not be updated yet
   int * host_dst = dst.getPointer(care::CPU, false);
   int * host_src = src.getPointer(care::CPU, false);

   for (int i = 0; i < 2; ++i) {
      if (i == 0) {
         printf("host_dst %p host_src %p\n", host_dst, host_src);
      }
      EXPECT_EQ(host_dst[i], -1);
      EXPECT_EQ(host_src[i], i);
   }

   packer->flushActions();

   care::gpuDeviceSynchronize();

#ifdef CARE_GPUCC
   // pack should have happened on the device, so
   // host data should not be updated yet
   for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(host_dst[i], -1);
      EXPECT_EQ(host_src[i], i);
   }
#endif

   // bringing stuff back to the host, dst[i] should now be i
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      EXPECT_EQ(dst[i], i);
      EXPECT_EQ(src[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   src.free();
   dst.free();
}

GPU_TEST(TestPacker, packFixedRangeMacro) {
   FUSIBLE_LOOPS_START

   int arrSize = 1024;
   care::host_device_ptr<int> src(arrSize);
   care::host_device_ptr<int> dst(arrSize);

   // initialize the src and dst on the host
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      src[i] = i;
      dst[i] = -1;
   } CARE_SEQUENTIAL_LOOP_END

   int pos = 0;

   {
      auto __fuser__ = LoopFuser::getInstance();
      auto __fusible_offset__ = __fuser__->getOffset();
      int scan_pos = 0;
      __fuser__->registerAction(0, arrSize, scan_pos,
                                [=] FUSIBLE_DEVICE(int, int *, int const*, int, fusible_registers){ },
                                [=] FUSIBLE_DEVICE(int i, int *, fusible_registers) {
         i += 0 -  __fusible_offset__ ;
         if (i < arrSize) {
            dst[pos+i] = src[i];
         }
         return 0;
      });
   }

   care::gpuDeviceSynchronize();

   // pack should  not have happened yet so
   // host data should not be updated yet
   int * host_dst = dst.getPointer(care::CPU, false);
   int * host_src = src.getPointer(care::CPU, false);

   for (int i = 0; i < arrSize; ++i) {
      EXPECT_EQ(host_dst[i], -1);
      EXPECT_EQ(host_src[i], i);
   }

   FUSIBLE_LOOPS_STOP

   // bringing stuff back to the host, dst[i] should now be i
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      EXPECT_EQ(dst[i], i);
      EXPECT_EQ(src[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   src.free();
   dst.free();
}

GPU_TEST(TestPacker, singleFusedLoop) {

   FUSIBLE_LOOPS_START
   chai::ManagedArray<int> test(2) ;
   FUSIBLE_LOOP_STREAM(i, 0, 2) {
      test[i] = 0 ;
   } FUSIBLE_LOOP_STREAM_END
   FUSIBLE_LOOPS_STOP

   CARE_SEQUENTIAL_LOOP(i,0,2) {
      EXPECT_EQ(test[i],0);
   } CARE_SEQUENTIAL_LOOP_END
   test.free();
}

GPU_TEST(TestPacker, fuseFixedRangeMacro) {
   FUSIBLE_LOOPS_START
   int arrSize = 8;
   care::host_device_ptr<int> src(arrSize);
   care::host_device_ptr<int> dst(arrSize);
   // initialize the src and dst on the host
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      src[i] = i;
      dst[i] = -1;
   } CARE_SEQUENTIAL_LOOP_END
   int pos = 0;
   FUSIBLE_LOOP_STREAM(i, 0, arrSize/2) {
      dst[pos+i] = src[i];
   } FUSIBLE_LOOP_STREAM_END
   pos += arrSize/2;
   FUSIBLE_LOOP_STREAM(i, 0, arrSize/2) {
      dst[pos+i] = src[i+pos]*2;
   } FUSIBLE_LOOP_STREAM_END

   care::gpuDeviceSynchronize();

   // pack should  not have happened yet so
   // host data should not be updated yet
   int * host_dst = dst.getPointer(care::CPU, false);
   int * host_src = src.getPointer(care::CPU, false);
   for (int i = 0; i < arrSize; ++i) {
      EXPECT_EQ(host_dst[i], -1);
      EXPECT_EQ(host_src[i], i);
   }
   FUSIBLE_LOOPS_STOP
   // bringing stuff back to the host, dst[i] should now be i
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize/2) {
      EXPECT_EQ(dst[i], i);
      EXPECT_EQ(src[i], i);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, arrSize/2, arrSize) {
      EXPECT_EQ(dst[i], i*2);
      EXPECT_EQ(src[i], i);
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(performanceWithoutPacker, allOfTheStreams) {
   int arrSize = 128;
   int timesteps = 5;
   care::host_device_ptr<int> src(arrSize);
   care::host_device_ptr<int> dst(arrSize);
   // initialize the src and dst on the device
   for (int t = 0; t < timesteps; ++t) {
      CARE_STREAM_LOOP(i, 0, arrSize) {
         src[i] = i;
         dst[i] = -1;
      } CARE_STREAM_LOOP_END
      for (int i = 0; i < arrSize; ++i) {
         CARE_STREAM_LOOP(j, i, i+1) {
            dst[j] = src[j];
         } CARE_STREAM_LOOP_END
      }
      // bringing stuff back to the host, dst[i] should now be i
      CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
         EXPECT_EQ(dst[i], i);
         EXPECT_EQ(src[i], i);
      } CARE_SEQUENTIAL_LOOP_END
   }
}


GPU_TEST(performanceWithPacker, allOfTheFuses) {
   int arrSize = 1;
   int timesteps = 5;
   care::host_device_ptr<int> src(arrSize);
   care::host_device_ptr<int> dst(arrSize);
   for (int t = 0; t < timesteps; ++t) {
      // initialize the src and dst on the device
      CARE_STREAM_LOOP(i, 0, arrSize) {
         src[i] = i;
         dst[i] = -1;
      } CARE_STREAM_LOOP_END
      FUSIBLE_LOOPS_START
      for (int i = 0; i < arrSize; ++i) {
         FUSIBLE_LOOP_STREAM(j, i, i+1) {
            dst[j] = src[j];
         } FUSIBLE_LOOP_STREAM_END
      }
      FUSIBLE_LOOPS_STOP
      // bringing stuff back to the host, dst[i] should now be i
      CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
         EXPECT_EQ(dst[i], i);
         EXPECT_EQ(src[i], i);
      } CARE_SEQUENTIAL_LOOP_END
   }
}


GPU_TEST(orderDependent, basic_test) {
   int arrSize = 16;
   int timesteps = 5;
   care::host_device_ptr<int> A(arrSize);
   care::host_device_ptr<int> B(arrSize);
   // initialize A on the host
   CARE_STREAM_LOOP(i, 0, arrSize) {
      A[i] = 0;
      B[i] = 0;
   } CARE_STREAM_LOOP_END
   // Note - this use to test PRESERVE_ORDER but that implementation was pretty flawed,
   // prefer to use PHASES instead. 
   FUSIBLE_LOOPS_START
   for (int t = 0; t < timesteps; ++t) {
      FUSIBLE_LOOP_PHASE(i, 0, arrSize, t) {
         if (t %2 == 0) {
            A[i] = (A[i] + 1)<<t;
         }
         else {
            A[i] = (A[i] + 1)>>t;
         }
      } FUSIBLE_LOOP_PHASE_END
      // do the same thing, but as separate kernels in a sequence
      CARE_STREAM_LOOP(i, 0, arrSize) {
         if (t %2 == 0) {
            B[i] = (B[i] + 1)<<t;
         }
         else {
            B[i] = (B[i] + 1)>>t;
         }
      } CARE_STREAM_LOOP_END
   }
   FUSIBLE_LOOPS_STOP
   // bringing stuff back to the host, A[i] should now be B[i]
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      EXPECT_EQ(A[i], B[i]);
   } CARE_SEQUENTIAL_LOOP_END
}
static
FUSIBLE_DEVICE bool printAndAssign(care::host_device_ptr<int> B, int i) {
   return B[i] == 1;
}

GPU_TEST(fusible_scan, basic_fusible_scan) {
   // arrSize should be even for this test
   int arrSize = 4;
   care::host_device_ptr<int> A(arrSize, "A");
   care::host_device_ptr<int> B(arrSize, "B");
   care::host_device_ptr<int> A_scan(arrSize, "A_scan");
   care::host_device_ptr<int> B_scan(arrSize, "B_scan");
   care::host_device_ptr<int> AB_scan(arrSize, "AB_scan");
   // initialize A on the host
   CARE_STREAM_LOOP(i, 0, arrSize) {
      A[i] = i%2;
      B[i] = (i+1)%2;
      A_scan[i] = 0;
      B_scan[i] = 0;
      AB_scan[i] = 0;
   } CARE_STREAM_LOOP_END

   FUSIBLE_LOOPS_START
   LoopFuser::getInstance()->setVerbose(true);

   int a_pos = 0;
   int b_pos = 0;
   int ab_pos = 0;
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, a_pos, A[i] == 1) {
      A_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, a_pos)
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, b_pos, printAndAssign(B, i)) {
      B_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, b_pos)
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, ab_pos, (A[i] == 1) || (B[i] ==1)) {
      AB_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, ab_pos)

   FUSIBLE_LOOPS_STOP
   LoopFuser::getInstance()->setVerbose(false);

   // sum up the results of the scans
   RAJAReduceSum<int> sumA(0);
   RAJAReduceSum<int> sumB(0);
   RAJAReduceSum<int> sumAB(0);
   CARE_STREAM_LOOP(i, 0, arrSize) {
      sumA += A_scan[i];
      sumB += B_scan[i];
      sumAB += AB_scan[i];
   } CARE_STREAM_LOOP_END

   // check sums
   EXPECT_EQ((int)sumA, arrSize/2);
   EXPECT_EQ((int)sumB, arrSize/2);
   EXPECT_EQ((int)sumAB, arrSize);
   // check scan positions
   EXPECT_EQ(a_pos, arrSize/2);
   EXPECT_EQ(b_pos, arrSize/2);
   EXPECT_EQ(ab_pos, arrSize);
}

GPU_TEST(fusible_dependent_scan, basic_dependent_fusible_scan) {
   // sarrSize should be even for this test
   int arrSize = 4;
   care::host_device_ptr<int> A(arrSize, "A");
   care::host_device_ptr<int> B(arrSize, "B");
   care::host_device_ptr<int> A_scan(3*arrSize, "A_scan");
   care::host_device_ptr<int> B_scan(3*arrSize, "B_scan");
   care::host_device_ptr<int> AB_scan(3*arrSize, "AB_scan");

   // initialize
   CARE_STREAM_LOOP(i, 0, arrSize) {
      A[i] = i%2;
      B[i] = (i+1)%2;
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(i, 0, 3*arrSize) {
      A_scan[i] = 0;
      B_scan[i] = 0;
      AB_scan[i] = 0;
   } CARE_STREAM_LOOP_END

   FUSIBLE_LOOPS_START

   int result_pos = 0;
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, A[i] == 1) {
      A_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, printAndAssign(B, i)) {
      B_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)
   FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, (A[i] == 1) || (B[i] ==1)) {
      AB_scan[pos] = 1;
   } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)

   FUSIBLE_LOOPS_STOP

   // sum up the results of the scans
   {
      RAJAReduceSum<int> sumA(0);
      RAJAReduceSum<int> sumB(0);
      RAJAReduceSum<int> sumAB(0);
      CARE_STREAM_LOOP(i, 0, 3*arrSize) {
         sumA += A_scan[i];
         sumB += B_scan[i];
         sumAB += AB_scan[i];
      } CARE_STREAM_LOOP_END

      // check sums
      EXPECT_EQ((int)sumA, arrSize/2);
      EXPECT_EQ((int)sumB, arrSize/2);
      EXPECT_EQ((int)sumAB, arrSize);
   }
   // sum up the results of the scans within the expected ranges of the scans
   {
      RAJAReduceSum<int> sumA(0);
      RAJAReduceSum<int> sumB(0);
      RAJAReduceSum<int> sumAB(0);
      CARE_STREAM_LOOP(i, 0, arrSize/2) {
         sumA += A_scan[i];
      } CARE_STREAM_LOOP_END
      CARE_STREAM_LOOP(i, arrSize/2, arrSize) {
         sumB += B_scan[i];
      } CARE_STREAM_LOOP_END
      CARE_STREAM_LOOP(i, arrSize, 2*arrSize) {
         sumAB += AB_scan[i];
      } CARE_STREAM_LOOP_END

      // check sums
      EXPECT_EQ((int)sumA, arrSize/2);
      EXPECT_EQ((int)sumB, arrSize/2);
      EXPECT_EQ((int)sumAB, arrSize);
   }
   // check scan positions
   EXPECT_EQ(result_pos, arrSize*2);
}

GPU_TEST(fusible_loops_and_scans, mix_and_match) {
   // should be even for this test
   int arrSize = 4;
   care::host_device_ptr<int> A(arrSize, "A");
   care::host_device_ptr<int> B(arrSize, "B");
   care::host_device_ptr<int> C(arrSize, "C");
   care::host_device_ptr<int> D(arrSize, "D");
   care::host_device_ptr<int> E(arrSize, "E");
   care::host_device_ptr<int> A_scan(3*arrSize, "A_scan");
   care::host_device_ptr<int> B_scan(3*arrSize, "B_scan");
   care::host_device_ptr<int> AB_scan(3*arrSize, "AB_scan");
   // initialize A on the host
   CARE_STREAM_LOOP(i, 0, arrSize) {
      A[i] = i%2;
      B[i] = (i+1)%2;
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(i, 0, 3*arrSize) {
      A_scan[i] = 0;
      B_scan[i] = 0;
      AB_scan[i] = 0;
   } CARE_STREAM_LOOP_END

   FUSIBLE_LOOPS_START
   FUSIBLE_LOOP_STREAM(i, 0, arrSize) {
      C[i] = i;
   } FUSIBLE_LOOP_STREAM_END

   int outer_dim = 2;
   care::host_ptr<int> results(outer_dim);
   for (int m = 0; m < outer_dim; ++m) {
      results[m] = 0;
      int & result_pos = results[m];

      FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, A[i] == 1) {
         A_scan[pos] = 1;
      } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)
      FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, printAndAssign(B, i)) {
         B_scan[pos] = 1;
      } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)
      FUSIBLE_LOOP_SCAN(i, 0, arrSize, pos, result_pos, (A[i] == 1) || (B[i] ==1)) {
         AB_scan[pos] = 1;
      } FUSIBLE_LOOP_SCAN_END(arrSize, pos, result_pos)
   }
   FUSIBLE_LOOP_STREAM(i, 0, arrSize) {
      D[i] = 2*i;
   } FUSIBLE_LOOP_STREAM_END
   FUSIBLE_LOOP_STREAM(i, 0, arrSize) {
      E[i] = 3*i;
   } FUSIBLE_LOOP_STREAM_END

   FUSIBLE_LOOPS_STOP

   // sum up the results of the scans
   {
      RAJAReduceSum<int> sumA(0);
      RAJAReduceSum<int> sumB(0);
      RAJAReduceSum<int> sumAB(0);
      CARE_STREAM_LOOP(i, 0, 3*arrSize) {
         sumA += A_scan[i];
         sumB += B_scan[i];
         sumAB += AB_scan[i];
      } CARE_STREAM_LOOP_END

      // check sums
      EXPECT_EQ((int)sumA, arrSize/2);
      EXPECT_EQ((int)sumB, arrSize/2);
      EXPECT_EQ((int)sumAB, arrSize);
   }
   // sum up the results of the scans within the expected ranges of the scans
   {
      RAJAReduceSum<int> sumA(0);
      RAJAReduceSum<int> sumB(0);
      RAJAReduceSum<int> sumAB(0);
      CARE_STREAM_LOOP(i, 0, arrSize/2) {
         sumA += A_scan[i];
      } CARE_STREAM_LOOP_END
      CARE_STREAM_LOOP(i, arrSize/2, arrSize) {
         sumB += B_scan[i];
      } CARE_STREAM_LOOP_END
      CARE_STREAM_LOOP(i, arrSize, 2*arrSize) {
         sumAB += AB_scan[i];
      } CARE_STREAM_LOOP_END

      // check sums
      EXPECT_EQ((int)sumA, arrSize/2);
      EXPECT_EQ((int)sumB, arrSize/2);
      EXPECT_EQ((int)sumAB, arrSize);
   }
   // check scan positions
   for (int m = 0; m < outer_dim; ++m) {
      EXPECT_EQ(results[m], arrSize*2);
   }

   // Check that FUSIBLE STREAM loops interwoven between scans also worked (yes, they had race conditions, but all races should have
   // written the same value)

   CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
      EXPECT_EQ(C[i], i);
      EXPECT_EQ(D[i], 2*i);
      EXPECT_EQ(E[i], 3*i);
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(fusible_scan_custom, basic_fusible_scan_custom) {
   // arrSize should be even for this test
   int arrSize = 4;
   care::host_device_ptr<int> A(arrSize, "A");
   care::host_device_ptr<int> B(arrSize, "B");
   care::host_device_ptr<int> AB_scan(arrSize*2, "AB_scan");
   // initialize A on the host
   CARE_STREAM_LOOP(i, 0, arrSize*2) {
      if (i / arrSize == 0) { 
         AB_scan[i] = 2;
      }
      else {
         AB_scan[i] = 3;
      }
   } CARE_STREAM_LOOP_END
   // convert to offsets
   exclusive_scan<int, RAJAExec>(AB_scan, nullptr, arrSize*2, RAJA::operators::plus<int>{}, 0, true);
   int offset = AB_scan.pick(arrSize);
   CARE_STREAM_LOOP(i,arrSize,arrSize*2) {
      AB_scan[i] -= offset;
   } CARE_STREAM_LOOP_END

   FUSIBLE_LOOPS_START
   LoopFuser::getInstance()->setVerbose(true);

   FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(i, 0, arrSize, A) {
      A[i] = 2;
   } FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(i,arrSize, A)

   FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(i, 0, arrSize, B) {
      B[i] = 3;
   } FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(i, arrSize, B)

   FUSIBLE_LOOPS_STOP
   LoopFuser::getInstance()->setVerbose(false);

   //  check answer
   CARE_SEQUENTIAL_LOOP(i, 0, arrSize*2) {
      int indx = i%arrSize;
      if (i / arrSize == 0 ) { 
         EXPECT_EQ(AB_scan[i],A[indx]);
      }
      else {
         EXPECT_EQ(AB_scan[i],B[indx]);
      }
   } CARE_SEQUENTIAL_LOOP_END

}

GPU_TEST(fusible_phase, fusible_loop_phase) {
   int arrSize = 128;
   int timesteps = 3;
   care::host_device_ptr<int> A1(arrSize);
   care::host_device_ptr<int> A2(arrSize);
   care::host_device_ptr<int> A3(arrSize);
   care::host_device_ptr<int> B1(arrSize);
   care::host_device_ptr<int> B2(arrSize);
   care::host_device_ptr<int> B3(arrSize);
   care::host_device_ptr<int> C1(arrSize);
   care::host_device_ptr<int> C2(arrSize);
   care::host_device_ptr<int> C3(arrSize);

   care::host_device_ptr<int> As[] = {A1,A2,A3};
   care::host_device_ptr<int> Bs[] = {B1,B2,B3};
   care::host_device_ptr<int> Cs[] = {C1,C2,C3};

   /* OK, so this is a bit convoluted, but the idea is to have a test where both execution order
    * matters, we shuffle which order we introduce kernels, and we have some memory as to whether
    * kernels were executed in the right order.
    * The A arrays are going to be written during phase execution, the B arrays should be written
    * in a Stream execution in the same order, and the C arrays are used to make sure the second
    * phase is happening after the first phase as a basic test of the phase scheduler.
    * */

   FUSIBLE_LOOPS_START
   for (int t = 0; t < timesteps; ++t) {
      care::host_device_ptr<int> A = As[t];
      care::host_device_ptr<int> B = Bs[t];
      care::host_device_ptr<int> C = Cs[t];


      CARE_STREAM_LOOP(i, 0, arrSize) {
         A[i] = -2;
         C[i] = -2;
         switch (t) {
            case 0:
               B[i] = -1;
               break;
            case 1:
               B[i] = 0;
               break;
            case 2:
               B[i] = -2;
               break;
         }
      } CARE_STREAM_LOOP_END

      if (t != 2) {
         FUSIBLE_LOOP_PHASE(i, 0, arrSize, __LINE__) {
            A[i] /= 2 ;
         } FUSIBLE_LOOP_PHASE_END
      }

      if (t == 1) {
         FUSIBLE_LOOP_PHASE(i, 0, arrSize, __LINE__) {
            /* only increment if the the /=2 actually happened and A[i] changed from -2 to -1 */
            if (A[i] == -1) {
               A[i] += 1 ;
            }
            C[i] = A[i];
         } FUSIBLE_LOOP_PHASE_END
      }

      FUSIBLE_LOOP_PHASE(i, 0, arrSize, __LINE__) {
         if (t %2 == 0) {
            A[i] = (A[i] + 1)<<t;
         }
         else {
            A[i] = (A[i] + 1)>>t;
         }
      } FUSIBLE_LOOP_PHASE_END
      // do the same thing, but as separate kernels in a sequence
      CARE_STREAM_LOOP(i, 0, arrSize) {
         if (t %2 == 0) {
            B[i] = (B[i] + 1)<<t;
         }
         else {
            B[i] = (B[i] + 1)>>t;
         }
      } CARE_STREAM_LOOP_END
#ifndef CARE_FUSIBLE_LOOPS_DISABLE
      // check that no phases have been executed yet
      CARE_SEQUENTIAL_LOOP(i, 0, arrSize) {
         EXPECT_EQ(A[i], -2);
         EXPECT_EQ(C[i], -2);
      } CARE_SEQUENTIAL_LOOP_END
#endif
      /* the captures and CHAI checks have already occurred, the FUSIBLE_LOOPS_STOP
       * won't update them, so we need to mark A and C as touched on the device
       * so we get fresh data after the flush */
      A.registerTouch(care::GPU);
      C.registerTouch(care::GPU);
      FUSIBLE_PHASE_RESET
   }
   FUSIBLE_LOOPS_STOP_ASYNC
   // bringing stuff back to the host, A[i] should now be B[i], C[i] should remember what A[i] was on the second phase.
   for (int t = 0; t < timesteps; ++t) {
      care::host_device_ptr<int> A = As[t];
      care::host_device_ptr<int> B = Bs[t];
      care::host_device_ptr<int> C = Cs[t];
      CARE_SEQUENTIAL_LOOP(i, 0, 5) {
         EXPECT_EQ(A[i], B[i]);
         if (t == 1) {
            EXPECT_EQ(C[i], 0);
         }
         else {
            EXPECT_EQ(C[i], -2);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

}
// TODO: FUSIBLE_LOOP_STREAM Should not batch if FUSIBLE_LOOPS_START has not been called.
// TODO: test with two START and STOP to make sure new stuff is overwriting the old stuff.
//
#endif
#endif // CARE_ENABLE_LOOP_FUSER

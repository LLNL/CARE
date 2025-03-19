//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SORT_FUSER_H
#define CARE_SORT_FUSER_H

#include "care/device_ptr.h"
#include "care/host_ptr.h"
#include "care/host_device_ptr.h"
#include "care/LoopFuser.h"
#include "care/algorithm.h"

namespace care {
   template <typename T>
   class SortFuser {

   public:
      SortFuser() = default;
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief resets the Fuser to be prepared for a new set of arrays. 
      ///        Must be called after any sort() / uniq()/ sortUniq() to free
      ///        temporary allocations.
      ///////////////////////////////////////////////////////////////////////////
      void reset(); 
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief adds an array to be sorted after a later call to sort()
      /// @param[in] array - an array to be sorted
      /// @param[in] len - the length of the array
      /// @param[in] range - the span of values in the array
      ///////////////////////////////////////////////////////////////////////////
      void fusibleSortArray(host_device_ptr<T> array, int len, T range);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief sorts all arrays registered with the Fuser via sortArray
      ///////////////////////////////////////////////////////////////////////////
      void sort();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief adds an array to be uniq-d after a later call to uniq()
      /// @param[in] array - an array to be sorted
      /// @param[in] len - the length of the array
      /// @param[in] range - the span of values in the array
      /// @param[out] the output of the uniq operation. (Valid after a call to uniq())
      /// @param[out] the number of uniq values. (Valid after a call to uniq())
      ///////////////////////////////////////////////////////////////////////////
      void fusibleUniqArray(host_device_ptr<T> array, int len, T range, host_device_ptr<T> & out_array, int &out_length);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief adds an array to be sorted and uniqued after a later call to sortUniq()
      /// @param[in] array - an array to be sorted and uniqued
      /// @param[in] len - the length of the array
      /// @param[in] range - the span of values in the array
      /// @param[out] the output of the uniq operation. (Valid after a call to sortUniq())
      /// @param[out] the number of uniq values. (Valid after a call to sortUniq())
      ///////////////////////////////////////////////////////////////////////////
      void fusibleSortUniqArray(host_device_ptr<T> array, int len, T range, host_device_ptr<T> & out_array, int &out_length);
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief uniq's all arrays registered with the Fuser via uniqArray or sortUniqArray
      /// @param isSorted - whether all arrays were sorted before the call to uniq.
      ///////////////////////////////////////////////////////////////////////////
      void uniq(bool isSorted=true, bool realloc=true);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief sorts and uniq's all arrays registered with the Fuser via uniqArray or sortUniqArray
      ///////////////////////////////////////////////////////////////////////////
      void sortUniq(bool realloc=true) { uniq(false, realloc);}

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief returns the intermediate result of the sorted/uniqued arrays
      ///        in a single bulk allocation, with the results concatanated
      ///        with each other.
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<T> getConcatenatedResult(); 
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief returns the intermediate lengths of the result of any uniqued arrays
      ///        in a single bulk allocation, with the lengths concatanated
      ///        with each other.
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<T> getConcatenatedLengths(); 

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief the default destructor
      ///////////////////////////////////////////////////////////////////////////
      ~SortFuser() = default;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief assembles currently registered arrays into a combined buffer, 
      ///        offsetting their values so that they own their own range of the
      ///        space in T.
      ///////////////////////////////////////////////////////////////////////////
#ifndef CARE_GPUCC      
   private:
#endif

      void assemble();
   protected:
      ///
      /// the arrays registered for sorting / uniqueing
      ///
      std::vector<host_device_ptr<T>> m_arrays_to_sort;
      ///
      /// the lengths of the arrays
      ///
      std::vector<int> m_lengths;
      ///
      /// the output arrays registered for uniqueing
      ///
      std::vector<host_device_ptr<T> *> m_out_arrays;
      ///
      /// the output lengths of registered for uniqueing
      ///
      std::vector<int *> m_out_lengths;
      ///
      /// the offsets of the arrays in the concatenated array
      ///
      std::vector<int> m_offsets;
      ///
      /// the number of arrays to sort
      ///
      int m_num_arrays;
      ///
      /// the total length of the arrays to sort (should be sum of m_lengths)
      ///
      int m_total_length;
      ///
      /// the max range enountered so far. This is what we use to give each
      /// array a unique range so that sorting does not mix entries from
      /// different arrays. If m_max_range*m_num_arrays approaches T_MAX,
      /// then it is possible to get bogus answers due to overflow errors.
      ///
      T m_max_range;
      ///
      /// The concatenated result - used as scratch space. There are use 
      /// cases where the user might want this, so it is exposed via
      /// getConcatenatedResult()
      ///
      host_device_ptr<T> m_concatenated_result;
      host_device_ptr<T> m_concatenated_lengths;
      

   };
   

   template <typename T>
   void SortFuser<T>::reset() {
      m_lengths.resize(0);
      m_offsets.resize(0);
      m_arrays_to_sort.resize(0);
      m_num_arrays = 0;
      m_total_length = 0;
      if (m_concatenated_result != nullptr) {
         m_concatenated_result.free();
      }
      m_concatenated_result = nullptr;
      if (m_concatenated_lengths != nullptr) {
         m_concatenated_lengths.free();
      }
      m_concatenated_lengths = nullptr;
      
      m_out_arrays.resize(0);
      m_out_lengths.resize(0);
   }


   template <typename T>
   host_device_ptr<T> SortFuser<T>::getConcatenatedResult() {
      return m_concatenated_result;
   }
   
   template <typename T>
   host_device_ptr<T> SortFuser<T>::getConcatenatedLengths() {
      return m_concatenated_lengths;
   }

   template <typename T>
   void SortFuser<T>::fusibleSortArray(host_device_ptr<T> array, int len, T range) {
      m_lengths.push_back(len);
      m_arrays_to_sort.push_back(array);
      m_offsets.push_back(m_total_length);
      ++m_num_arrays;
      m_total_length += len;
      m_max_range = care::max(m_max_range,range);
   }
   
   template <typename T>
   void SortFuser<T>::fusibleUniqArray(host_device_ptr<T> array, int len, T range,
                                host_device_ptr<T> &out_array, int &out_len) {
      m_lengths.push_back(len);
      m_arrays_to_sort.push_back(array);
      m_offsets.push_back(m_total_length);
      ++m_num_arrays;
      m_total_length += len;
      m_max_range = care::max(m_max_range,range);
      m_out_arrays.push_back(&out_array); 
      m_out_lengths.push_back(&out_len);
   }
   
   template <typename T>
   void SortFuser<T>::fusibleSortUniqArray(host_device_ptr<T> array, int len, T range,
                                host_device_ptr<T> &out_array, int &out_len) {
      // sort uniqs require the same metadata / output as a uniq
      fusibleUniqArray(array,len,range,out_array,out_len);
   }
   
   template <typename T>
   void SortFuser<T>::assemble() {
      if (m_concatenated_result != nullptr) {
         m_concatenated_result.free();
      }
      m_concatenated_result = host_device_ptr<T>(m_total_length);
      FUSIBLE_LOOPS_START
      for (int a = 0; a < m_num_arrays; ++a) {
          host_device_ptr<T> array = m_arrays_to_sort[a];
          host_device_ptr<T> result = m_concatenated_result;
          int offset = m_offsets[a];
          T max_range = m_max_range;
          FUSIBLE_LOOP_STREAM(i,0,m_lengths[a]) {
             result[i+offset] = array[i] + max_range*a;
          } FUSIBLE_LOOP_STREAM_END
      }
      FUSIBLE_LOOPS_STOP
   }

   ///
   /// perform a fused sort
   /// 
   template <typename T>
   void SortFuser<T>::sort() {
      assemble();
      care::sortArray(RAJAExec{}, m_concatenated_result, m_total_length);
      // scatter answer back into original arrays by subtracting off the range
      // multipliers
      FUSIBLE_LOOPS_START
      for (int a = 0; a < m_num_arrays; ++a) {
          host_device_ptr<T> array = m_arrays_to_sort[a];
          host_device_ptr<T> result = m_concatenated_result;
          int offset = m_offsets[a];
          T max_range = m_max_range;
          FUSIBLE_LOOP_STREAM(i,0,m_lengths[a]) {
             result[i+offset] -= max_range*a;
             array[i] = result[i+offset];
          } FUSIBLE_LOOP_STREAM_END
      }
      FUSIBLE_LOOPS_STOP
   }

   ///
   /// perform a fused uniq, sorting if necessary.
   ///
   template <typename T>
   void SortFuser<T>::uniq(bool isSorted, bool realloc) {
      assemble();
      host_device_ptr<T> concatenated_out;
      if (!isSorted) {
         care::sortArray(RAJAExec{}, m_concatenated_result, m_total_length);
      }
      
      // do the unique of the concatenated sort result
      int outLen;
      care::uniqArray(RAJAExec{}, reinterpret_cast<host_device_ptr<const T>&>(m_concatenated_result), m_total_length, concatenated_out, outLen);
      
      /// determine new offsets by looking for boundaries in max_range
      host_device_ptr<int> out_offsets(m_num_arrays+1, "out_offsets");

      int num_arrays = m_num_arrays;
      T max_range = m_max_range;
      CARE_STREAM_LOOP(i,0,outLen+1) {
         int prev_array = i == 0 ? -1 :  concatenated_out[i-1] / max_range;
         int next_array = i == outLen ? num_arrays : concatenated_out[i] / max_range;
         if (prev_array != next_array) {
            // we are at a boundary
            for (int j = prev_array+1; j <= next_array; ++j) {
               out_offsets[j] = i;
            }
         }
      } CARE_STREAM_LOOP_END
      host_ptr<const int> host_out_offsets = out_offsets;
      host_device_ptr<int> concatenated_lengths(m_num_arrays, "concatenated_lengths");
      host_device_ptr<T> result = concatenated_out;
      
      // set up a 2D kernel, put per-array meta-data in pinned memory to eliminate cudaMemcpy's of the smaller dimension of data
      host_device_ptr<int> lengths(chai::ManagedArray<int>(m_num_arrays, chai::ZERO_COPY));
      host_device_ptr<host_device_ptr<int> > out_arrays(chai::ManagedArray<host_device_ptr<int>>(m_num_arrays, chai::ZERO_COPY));
      host_ptr<int> pinned_lengths = lengths.data(care::ZERO_COPY, false);
      host_ptr<host_device_ptr<int>>  pinned_out_arrays = out_arrays.data(care::ZERO_COPY, false);
      // initialized lengths, maxLength, and array of arrays for the 2D kernel
      int maxLength = 0;
      for (int a = 0; a < m_num_arrays; ++a ) {
         // update output length by doing subtraction of the offsets
         int & len = *m_out_lengths[a];
         len = host_out_offsets[a+1]-host_out_offsets[a];
         pinned_lengths[a]= len;
         maxLength = care::max(len, maxLength);
         if (realloc) {
            m_out_arrays[a]->realloc(len);
         }
         pinned_out_arrays[a] = *m_out_arrays[a];
      }
      // subtract out the offset, copy the result into individual arrays
      // (use of device pointer is to avoid clang-query rules that prevent capture of raw pointer)
      device_ptr<int> dev_pinned_lengths = lengths.data(ZERO_COPY, false);
      CARE_LOOP_2D_STREAM_JAGGED(i, 0, maxLength, lengths, a, 0, m_num_arrays, iFlattened)  {
         result[i+out_offsets[a]] -= max_range*a;
         out_arrays[a][i] = result[i+out_offsets[a]];
         if (i == 0) {
            concatenated_lengths[a] = dev_pinned_lengths[a];
         }
      } CARE_LOOP_2D_STREAM_JAGGED_END

      // m_concatenated_result contains result of the initial contcatenation, need to swap it
      // out with the result of the uniq
      m_concatenated_result.free();
      m_concatenated_result = concatenated_out;
      m_concatenated_lengths = concatenated_lengths;

      out_offsets.free();
      
      lengths.free();
      out_arrays.free();
   }
}

#endif // CARE_SORT_FUSER_H

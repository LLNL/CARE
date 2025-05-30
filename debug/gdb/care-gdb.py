##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################


import gdb.printing
import itertools


class HostDevicePtrPrinter:
   """Print a host_device_ptr object."""
   
   def __init__(self, val):
      self.val = val
      
   def to_string(self):
      num_elements = self.val['m_elems']

      if num_elements > 0:
         return "{{ size = {0} }}".format(num_elements)
      else:
         return "nullptr"

   def display_hint(self):
      return 'array'

   def children (self):
      for i in range(self.val['m_elems']):
         yield "[{0}]".format(i), self.val['m_active_pointer'][i]


def build_pretty_printer():
   pp = gdb.printing.RegexpCollectionPrettyPrinter("care")
   pp.add_printer('host_device_ptr', 'host_device_ptr', HostDevicePtrPrinter)
   return pp


gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())


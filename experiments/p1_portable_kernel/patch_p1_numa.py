"""P1: NUMA page placement becomes optional. Without numaif.h (WSL without libnuma-dev, macOS, Windows), move_pages is
a stub that places nothing and reports every page as unplaced; placement only moves pages, never changes arithmetic."""
import sys

p = sys.argv[1]
s = open(p).read()
a = "#include <numaif.h>\n"
b = """#if defined(__linux__) && defined(__has_include)
#if __has_include(<numaif.h>)
#include <numaif.h>
#define VH_HAVE_NUMA 1
#endif
#endif
#ifndef VH_HAVE_NUMA
#define VH_HAVE_NUMA 0
#define MPOL_MF_MOVE 2
// No NUMA API on this platform: nothing is migrated, every page reports "unknown node" (-1).
static inline long move_pages(int, unsigned long count, void** pages, const int*, int* status, int) {
  (void)pages;
  for (unsigned long i = 0; i < count; i++) status[i] = -1;
  return -1;
}
#endif
"""
assert s.count(a) == 1
s = s.replace(a, b)
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n",
              "bool have_numa() { return VH_HAVE_NUMA != 0; }\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
              "  m.def(\"have_numa\", &have_numa);\n", 1)
open(p, "w").write(s)
print("numa optional")

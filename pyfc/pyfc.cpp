//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "pyfc.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pyfc_ARRAY_API
#include <numpy/arrayobject.h>

void init_pyfc()
{
	import_array();
}

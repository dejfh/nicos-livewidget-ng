#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pytomo_ARRAY_API
#include <numpy/arrayobject.h>

void init_tomo()
{
	import_array();
}
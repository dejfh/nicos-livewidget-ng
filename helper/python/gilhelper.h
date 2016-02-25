#ifndef HELPER_PYTHON_GILHELPER_H
#define HELPER_PYTHON_GILHELPER_H

#include <Python.h>

namespace hlp
{
namespace python
{

struct EnsureGil {
    PyGILState_STATE gilState;
    EnsureGil()
    {
        gilState = PyGILState_Ensure();
    }
    ~EnsureGil()
    {
        PyGILState_Release(gilState);
    }
};

} // namespace python

} // namespace hlp

#endif // HELPER_PYTHON_GILHELPER_H

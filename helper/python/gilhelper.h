#ifndef HELPER_PYTHON_GILHELPER_H
#define HELPER_PYTHON_GILHELPER_H

#include <Python.h>
#include <cstddef>

namespace hlp
{
namespace python
{

struct Gil {
    PyGILState_STATE gilState;
    Gil()
    {
        gilState = PyGILState_Ensure();
    }
    ~Gil()
    {
        PyGILState_Release(gilState);
    }
};

struct Ref {
    PyObject *ptr;

    Ref()
        : ptr(nullptr)
    {
    }
    Ref(std::nullptr_t)
        : ptr(nullptr)
    {
    }
    Ref(PyObject *ptr, bool inc = false)
        : ptr(ptr)
    {
        if (inc)
            Py_XINCREF(ptr);
    }
    Ref(const Ref &other)
        : ptr(other.ptr)
    {
        Py_XINCREF(ptr);
    }
    Ref(Ref &&other)
        : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    ~Ref()
    {
        Py_XDECREF(ptr);
    }
    PyObject *steal()
    {
        PyObject *result = ptr;
        ptr = nullptr;
        return result;
    }
    PyObject *inc()
    {
        Py_XINCREF(ptr);
        return ptr;
    }
    void release()
    {
        Py_XDECREF(ptr);
        ptr = nullptr;
    }
    Ref &operator=(const Ref &other)
    {
        if (ptr == other.ptr)
            return *this;
        Py_XINCREF(other.ptr);
        Py_XDECREF(ptr);
        ptr = other.ptr;
        return *this;
    }
    Ref &operator=(Ref &&other)
    {
        Py_XDECREF(ptr);
        ptr = other.steal();
        return *this;
    }
    Ref &operator=(PyObject *other)
    {
        if (ptr == other)
            return *this;
        Py_XINCREF(other);
        Py_XDECREF(ptr);
        ptr = other;
        return *this;
    }
    operator bool() const
    {
        return ptr;
    }
    void swap(Ref &other)
    {
        std::swap(this->ptr, other.ptr);
    }
};

} // namespace python

} // namespace hlp

namespace std
{
inline void swap(hlp::python::Ref &a, hlp::python::Ref &b)
{
    a.swap(b);
}
} // namespace std

#endif // HELPER_PYTHON_GILHELPER_H

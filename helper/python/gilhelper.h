#ifndef HELPER_PYTHON_GILHELPER_H
#define HELPER_PYTHON_GILHELPER_H

#include <Python.h>
#include <cstddef>
#include <utility>

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
    Gil(const Gil &) = delete;
    Gil(Gil &&) = delete;

    ~Gil()
    {
        PyGILState_Release(gilState);
    }
};

/**
 * @brief Reference to a python object, automatically increasing and decreasing refcounts.
 *
 * This structure alwas owns the reference it contains.
 * To use Ref as an argument for a borrowing parameter use the value fo the @ref ptr field.
 * To use Ref as an arguemtn for a stealing parameter either use @ref inc() to increase the refcount and keep a reference,
 * or use @ref steal() to transfer the ownership and reset the Ref.
 */
struct Ref {
    PyObject *ptr;

    Ref(std::nullptr_t = nullptr)
        : ptr(nullptr)
    {
    }
    /**
     * @brief Ref Constructor. Initializes referencing to a given ptr.
     * @param inc If true, increases recount. Defaults to false.
     */
    Ref(PyObject *ptr, bool inc = false)
        : ptr(ptr)
    {
        if (inc)
            Py_XINCREF(ptr);
    }
    /**
     * @brief release Reduces refcount.
     */
    void release()
    {
        Py_XDECREF(ptr);
        ptr = nullptr;
    }
    /**
     * @brief ~Ref Destructor. Reduces refcount.
     */
    ~Ref()
    {
        release();
    }

    /**
     * @brief steal Resets ptr to null and returns the previous value. Leaves refcount unchanged.
     * @return
     */
    PyObject *steal()
    {
        PyObject *result = ptr;
        ptr = nullptr;
        return result;
    }
    /**
     * @brief inc
     * @return
     */
    PyObject *inc() const
    {
        return Ref(*this).steal();
    }

    /**
     * @brief Ref Move constructor. Steals from other. Leaves refcount unchanged.
     */
    Ref(Ref &&other)
        : ptr(other.steal())
    {
    }
    /**
     * @brief Ref Copy constructor. Copies from other. Increases refcount.
     */
    Ref(const Ref &other)
        : ptr(other.ptr)
    {
        Py_XINCREF(ptr);
    }
    /**
     * @brief operator = Move assignment. Steals from other. Reduces refcount of old ptr and leaves assigned refcount unchagned.
     */
    Ref &operator=(Ref &&other)
    {
        release();
        ptr = other.steal();
        return *this;
    }
    /**
     * @brief operator = Copy assignment. Copies from other. Increases refcount of copied ptr and reduces refcount of old ptr.
     */
    Ref &operator=(const Ref &other)
    {
        if (ptr == other.ptr)
            return *this;
        *this = Ref(other);
        return *this;
    }

    Ref &reset(PyObject *ptr = nullptr, bool inc = false)
    {
        if (this->ptr == ptr && inc)
            return *this;
        *this = Ref(ptr, inc);
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

#ifndef NDIM_NUMPY_H
#define NDIM_NUMPY_H

#include <Python.h>

#include <numpy/ndarrayobject.h>

#include "ndim/pointer.h"

#include "helper/python/gilhelper.h"

namespace ndim
{

inline hlp::python::Ref makePyArrayRef(const ndim::PointerVar<float> &ptr, const hlp::python::Gil & = hlp::python::Gil())
{
    std::vector<npy_intp> dims(ptr.shape.cbegin(), ptr.shape.cend());
    std::vector<npy_intp> strides;
    std::transform(ptr.strides.cbegin(), ptr.strides.cend(), std::back_inserter(strides), [](hlp::byte_offset_t stride) { return stride.value; });
    return PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT), ptr.dimensionality(), dims.data(), strides.data(),
        const_cast<float *>(ptr.data), NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, nullptr);
}

inline hlp::python::Ref makePyArrayRef(const ndim::PointerVar<const float> &ptr, const hlp::python::Gil & = hlp::python::Gil())
{
    std::vector<npy_intp> dims(ptr.shape.cbegin(), ptr.shape.cend());
    std::vector<npy_intp> strides;
    std::transform(ptr.strides.cbegin(), ptr.strides.cend(), std::back_inserter(strides), [](hlp::byte_offset_t stride) { return stride.value; });
    return PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT), ptr.dimensionality(), dims.data(), strides.data(),
        const_cast<float *>(ptr.data), NPY_ARRAY_ALIGNED, nullptr);
}

inline ndim::ShapeVar getShapeVar(PyArrayObject *array)
{
    size_t dimensionality = PyArray_NDIM(array);

    npy_intp *arrayShape = PyArray_SHAPE(array);
    ndim::ShapeVar shape(dimensionality);
    std::copy(arrayShape, arrayShape + dimensionality, shape.begin());

    return shape;
}

inline ndim::StridesVar getStridesVar(PyArrayObject *array)
{
    size_t dimensionality = PyArray_NDIM(array);

    npy_intp *arrayStrides = PyArray_STRIDES(array);
    ndim::StridesVar strides(dimensionality);
    std::transform(arrayStrides, arrayStrides + dimensionality, strides.begin(), [](npy_intp v) { return hlp::byte_offset_t(v); });

    return strides;
}

template <size_t Dimensionality>
ndim::Sizes<Dimensionality> getShape(PyArrayObject *array)
{
    if (Dimensionality != PyArray_NDIM(array))
        throw std::out_of_range("Dimensionality mismatch.");

    npy_intp *arrayShape = PyArray_SHAPE(array);
    ndim::Sizes<Dimensionality> shape;
    std::copy(arrayShape, arrayShape + Dimensionality, shape.begin());

    return shape;
}

template <size_t Dimensionality>
ndim::Strides<Dimensionality> getStrides(PyArrayObject *array)
{
    if (Dimensionality != PyArray_NDIM(array))
        throw std::out_of_range("Dimensionality mismatch.");

    npy_intp *arrayStrides = PyArray_STRIDES(array);
    ndim::Strides<Dimensionality> strides;
    std::transform(arrayStrides, arrayStrides + Dimensionality, strides.begin(), [](npy_intp v) { return hlp::byte_offset_t(v); });

    return strides;
}

} // namespace ndim

#endif // NDIM_NUMPY_H

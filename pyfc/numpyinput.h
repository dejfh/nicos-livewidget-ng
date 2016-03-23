#ifndef NUMPYINPUT
#define NUMPYINPUT

#include "fc/datafilterbase.h"

#include <Python.h>

#include <numpy/arrayobject.h>

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "helper/python/gilhelper.h"

template <size_t _Dimensionality>
class NumpyInput : public fc::FilterBase, virtual public fc::DataFilter<float, _Dimensionality>
{
public:
	using ElementType = float;
	static const size_t Dimensionality = _Dimensionality;

private:
	hlp::Threadsafe<PyObject *> m_data;
	mutable PyObject *m_current;

public:
	~NumpyInput()
	{
		PyObject *data = m_data.unguarded();
		if (data == nullptr && m_current == nullptr)
			return;

		hlp::python::EnsureGil gil;

		Py_XDECREF(data);
		Py_XDECREF(m_current);
	}

	PyObject *data() const
	{
		auto guard = m_data.lockConst();
		PyObject *data = guard.data();
		if (data == nullptr)
			return nullptr;

		hlp::python::EnsureGil gil;

		Py_INCREF(data);
		return data;
	}

	void setData(PyObject *data)
	{
		hlp::python::EnsureGil gil;

		data = PyArray_FromAny(data, PyArray_DescrFromType(NPY_FLOAT), Dimensionality, Dimensionality, NPY_ARRAY_ALIGNED, nullptr);

		PyObject *toDecrease = nullptr;

		{
			auto guard = m_data.lock();
			toDecrease = guard.data();
			guard.data() = data;
		}

		Py_XDECREF(toDecrease);
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(fc::PreparationProgress &progress) const override
	{
		{
			hlp::python::EnsureGil gil;
			Py_XDECREF(m_current); // Decrease outside of lock! May block.
			{
				auto guard = m_data.lockConst();
				m_current = guard.data();
				Py_XINCREF(m_current); // Increase inside of lock! Ref may decrease immediately after unlock. Incref does not block.
			}
		}
		progress.throwIfCancelled();
		hlp::notNull(m_current);

		auto array = hlp::cast_over_void<PyArrayObject *>(m_current);

		ndim::Sizes<Dimensionality> sizes;
		std::copy_n(PyArray_SHAPE(array), Dimensionality, sizes.begin());
		return sizes;
	}

	virtual fc::Container<ElementType, Dimensionality> getData(
		fc::ValidationProgress &progress, fc::Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(progress, recycle);

		auto array = hlp::cast_over_void<PyArrayObject *>(m_current);
		ndim::Sizes<Dimensionality> shape;
		std::copy_n(PyArray_SHAPE(array), Dimensionality, shape.begin());
		ndim::Strides<Dimensionality> byte_strides;

		npy_intp *strides = PyArray_STRIDES(array);
		std::transform(strides, strides + Dimensionality, byte_strides.begin(), [](npy_intp v) { return hlp::byte_offset_t(v); });
		ElementType *data = static_cast<ElementType *>(PyArray_DATA(array));

		ndim::pointer<const ElementType, Dimensionality> ptr(data, shape, byte_strides);
		return fc::makeConstRefContainer(ptr);
	}
};

#endif // NUMPYINPUT

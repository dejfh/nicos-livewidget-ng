#ifndef PYFC_NUMPYINPUT_H
#define PYFC_NUMPYINPUT_H

#include "fc/datafilterbase.h"

#include <Python.h>

#include "numpy.h"

#include "ndim/numpy.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "helper/python/gilhelper.h"

#include <iostream>

namespace pyfc {

template <size_t _Dimensionality>
class NumpyInput : public fc::FilterBase, virtual public fc::DataFilter<float, _Dimensionality>
{
public:
	using ElementType = float;
	static const size_t Dimensionality = _Dimensionality;

private:
	hlp::Threadsafe<hlp::python::Ref> m_data;
	mutable hlp::python::Ref m_current;

public:
	~NumpyInput()
	{
		hlp::python::Ref &data = m_data.unguardedMutable();
		if (!data && !m_current)
			return;

		hlp::python::Gil gil;
		hlp::unused(gil);

		data.release();
		m_current.release();
	}

	hlp::python::Ref data() const
	{
		auto &data = m_data.unguarded();

		if (data)
			return nullptr;

		hlp::python::Gil gil;
		hlp::unused(gil);

		return data;
	}

	void setData(const hlp::python::Ref &data, const hlp::python::Gil & = hlp::python::Gil())
	{
		hlp::python::Ref array =
			PyArray_FromAny(data.ptr, PyArray_DescrFromType(NPY_FLOAT), Dimensionality, Dimensionality, NPY_ARRAY_ALIGNED, nullptr);

		{
			auto guard = m_data.lock();
			std::swap(array, guard.data()); // Don't decrease refcount with active data lock. Destructor may be slow.
		}

		// Old guard.data will be dereferenced here.
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(fc::PreparationProgress &progress) const override
	{
		{
			hlp::python::Gil gil;
			hlp::unused(gil);

			m_current.release(); // Decreasing refcount may be slow. Do outside lock!
			{
				auto guard = m_data.lockConst();
				m_current = guard.data(); // Increasing refcount is fast. Is okay inside lock.
			}
		}
		progress.throwIfCancelled();
		hlp::throwIfNull(m_current.ptr);

		auto array = hlp::cast_over_void<PyArrayObject *>(m_current.ptr);

		return ndim::getShape<Dimensionality>(array);
	}

	virtual ndim::Container<ElementType, Dimensionality> getData(
		fc::ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(progress, recycle);

		auto array = hlp::cast_over_void<PyArrayObject *>(m_current.ptr);

		auto shape = ndim::getShape<Dimensionality>(array);
		auto strides = ndim::getStrides<Dimensionality>(array);

		ElementType *data = static_cast<ElementType *>(PyArray_DATA(array));

		ndim::pointer<const ElementType, Dimensionality> ptr(data, shape, strides);
		return ptr;
	}
};

} // namespace pyfc

#endif // PYFC_NUMPYINPUT_H

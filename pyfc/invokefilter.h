#ifndef PYFC_PYINVOKEFILTER_H
#define PYFC_PYINVOKEFILTER_H

#include <memory>
#include <vector>
#include <string>

#include <QVector>

#include "Python.h"

#include "pyfc/numpy.h"

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "ndim/pointer.h"
#include "ndim/numpy.h"
#include "ndim/iterator.h"

#include "helper/python/gilhelper.h"

#include <iostream>

namespace pyfc
{

class InvokeFilter : public fc::DataFilterVar<float>, public fc::FilterBase
{
public:
	using ElementType = float;

private:
	struct Config {
		QVector<std::shared_ptr<const fc::DataFilterVar<float>>> predecessors;
		hlp::python::Ref prepareProc;
		hlp::python::Ref getDataProc;

		void release()
		{
			prepareProc.release();
			getDataProc.release();
		}
	};

	hlp::Threadsafe<Config> m_config;

public:
	InvokeFilter() = default;
	~InvokeFilter()
	{
		hlp::python::Gil gil;
		hlp::unused(gil);

		Config &config = m_config.unguardedMutable(); // No lock necessary during destruction
		config.prepareProc.release();
		config.getDataProc.release();
	}

public:
	void setPredecessors(QVector<std::shared_ptr<const fc::DataFilterVar<float>>> predecessors)
	{
		this->invalidate();
		auto guard = m_config.lock();
		this->unregisterAsSuccessor(guard->predecessors);
		predecessors.swap(
			guard->predecessors); // Order of destruction ensures that teferences to old predecessors are released after the lock has been released.
		this->registerAsSuccessor(guard->predecessors);
	}
	void setTarget(hlp::python::Ref prepareProc, hlp::python::Ref getDataProc)
	{
		this->invalidate();
		auto guard = m_config.lock();
		std::swap(prepareProc, guard->prepareProc);
		std::swap(getDataProc, guard->getDataProc);
	}

	// DataFilterVar interface
public:
	virtual ndim::ShapeVar prepareVar(fc::PreparationProgress &progress) const override
	{
		hlp::python::Gil gil;
		hlp::unused(gil);

		auto config = m_config.get();
		progress.throwIfCancelled();

		hlp::python::Ref arglist = PyTuple_New(config.predecessors.size());

		size_t i = 0;
		for (const auto &predecessor : config.predecessors) {
			ndim::ShapeVar shape = predecessor->prepareVar(progress);

			hlp::python::Ref list = PyList_New(shape.size());
			for (size_t u = 0, uu = shape.size(); u < uu; ++u)
				PyList_SetItem(list.ptr, u, PyInt_FromSize_t(shape[u]));

			PyTuple_SetItem(arglist.ptr, i++, list.steal());
		}

		hlp::python::Ref result = PyObject_CallObject(config.prepareProc.ptr, arglist.ptr);

		ndim::ShapeVar shape(PyList_Size(result.ptr));
		for (size_t i = 0, ii = shape.size(); i < ii; ++i)
			shape[i] = PyInt_AsSsize_t(PyList_GetItem(result.ptr, i));

		return shape;
	}
	virtual ndim::ContainerVar<ElementType> getDataVar(fc::ValidationProgress &progress, ndim::ContainerBase<ElementType> *recycle) const override
	{
		hlp::python::Gil gil;
		hlp::unused(gil);

		auto printer = [](const ndim::ContainerVar<float> &container, PyArrayObject *array) {
			ndim::PointerVar<const float> ptr = container.constData();
			size_t D = ptr.shape.size();
			assert(D == 2);
			ndim::IndicesVar i(D, 0);
			std::vector<npy_intp> ai(D, 0);
			for (i[1] = 0; i[1] < ptr.shape[1]; ++i[1]) {
				ai[1] = i[1];
				for (i[0] = 0; i[0] < ptr.shape[0]; ++i[0]) {
					ai[0] = i[0];
					std::cout << '(' << i[0] << ", " << i[1] << ") : ";
					std::cout << ptr[i] << " / " << *(float*)PyArray_GetPtr(array, ai.data()) << std::endl;
				}
			}
		};

		auto config = m_config.get();
		progress.throwIfCancelled();

		hlp::python::Ref arglist = PyTuple_New(config.predecessors.size());

		size_t i = 0;
		for (const auto &predecessor : config.predecessors) {
			ndim::ContainerVar<float> data = predecessor->getDataVar(progress);
			hlp::python::Ref array = ndim::makePyArrayRef(data.constData(), gil);
			printer(data, (PyArrayObject *)(array.ptr));
			PyTuple_SetItem(arglist.ptr, i++, array.steal());
		}

		hlp::python::Ref result = PyObject_CallObject(config.getDataProc.ptr, arglist.ptr);

		result = PyArray_FromAny(result.ptr, PyArray_DescrFromType(NPY_FLOAT), 0, 0, NPY_ARRAY_ALIGNED, nullptr);
		PyArrayObject *array = (PyArrayObject *)result.ptr;

		ndim::ContainerVar<ElementType> container;
		if (recycle)
			container.swapOwnership(*recycle);
		container.resize(ndim::getShapeVar(array));

		hlp::python::Ref containerRef = ndim::makePyArrayRef(container.mutableData(), gil);
		PyArray_CopyInto((PyArrayObject *)containerRef.ptr, array);

		printer(container, (PyArrayObject *)(array));

		return container;
	}
};

} // namespace pyfc

#endif // PYFC_PYINVOKEFILTER_H

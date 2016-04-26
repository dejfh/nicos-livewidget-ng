#ifndef PYFC_PYINVOKEFILTER_H
#define PYFC_PYINVOKEFILTER_H

#include <memory>
#include <vector>

#include <QVector>

#include "Python.h"

#include "numpy/ndarrayobject.h"

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "pyfilter.h"

#include "ndim/pointer.h"

namespace pyfc {

class InvokeTarget {
public:
    virtual ~InvokeTarget() = default;
    virtual QVector<int> prepare(QVector<int> shape) = 0;
    virtual PyObject *getData(PyObject *sources) = 0;
};

template <size_t _Dimensionality>
class PyInvokeFilter : public fc::DataFilter<float, _Dimensionality>, public fc::FilterBase
{
public:
    size_t Dimensionality = _Dimensionality;

protected:
    hlp::Threadsafe<std::vector<std::shared_ptr<fc::DataFilterVar<float>>>> m_sources;
    std::unique_ptr<InvokeTarget> m_invokeTarget;

    QVector<int> invokePrepare(fc::PreparationProgress &progress) {
        auto guard = m_sources.lock();
        for (const auto &source : guard.data())
            source->prepareVar(progress);
        // TODO
    }
    PyObject *invokeGetData(fc::ValidationProgress &progress) {
        auto guard = m_sources.lock();
        for (const auto &source : guard.data())
            source->getDataVar(progress);
        // TODO
    }

public:
    void addPredecessor(std::unique_ptr<PySource> source) {
        auto guard = m_sources.lock();
        guard->push_back(std::move(source));
        guard->back()->addSuccessor(this->shared_from_this());
    }

    void clearPredecessors() {
        auto guard = m_sources.lock();
        for (const auto &predecessor : guard.data())
            predecessor->removeSuccessor(this);
        guard->clear();
    }

    // DataFilter interface
public:
    virtual ndim::sizes<Dimensionality> prepare(fc::PreparationProgress &progress) const override
    {
    }
    virtual ndim::Container<float, Dimensionality> getData(fc::ValidationProgress &progress, ndim::Container<float, Dimensionality> *recycle) const override
    {
    }
};

template <size_t _Dimensionality>
class PyInvokeFilterNd : public PyFilterNd<_Dimensionality>
{
public:
    static const size_t Dimensionality = _Dimensionality;

private:
    std::shared_ptr<PyInvokeFilter> m_filter;

public:
    PyInvokeFilterNd() = default;

    void clearPredecessors() {
        m_filter->clearPredecessors();
    }

    void addPredecessor(const FilterVar &filter) {
        m_filter->addPredecessor(makeSource(filter));
    }
};

} // namespace pyfc

using InvokeFilter1d = pyfc::PyInvokeFilterNd<1>;
using InvokeFilter2d = pyfc::PyInvokeFilterNd<2>;
using InvokeFilter3d = pyfc::PyInvokeFilterNd<3>;
using InvokeFilter4d = pyfc::PyInvokeFilterNd<4>;

#endif // PYFC_PYINVOKEFILTER_H

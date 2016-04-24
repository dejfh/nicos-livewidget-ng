#ifndef PYFC_PYINVOKEFILTER_H
#define PYFC_PYINVOKEFILTER_H

#include <memory>
#include <vector>

#include <QVector>

#include "Python.h"

#include "numpy/ndarrayobject.h"

#include "fc/datafilter.h"

#include "pyfilter.h"

namespace pyfc {

struct PySource {
    virtual ~PySource() = default;

    virtual QVector<int> prepare(PreparationProgress &progress) = 0;
    virtual PyObject *getData(ValidationProgress &progress) = 0;

    virtual void registerSuccessor(std::weak_ptr<Successor> successor) = 0;
    virtual void unregisterSuccessor(Successor *successor) = 0;
};

template <size_t _Dimensionality>
class PyDataFilterSource : public PySource
{
public:
    static const size_t Dimensionality = _Dimensionality;

private:
    std::shared_ptr<fc::DataFilter<float, Dimensionality>> m_predecessor;

public:
    PyDataFilterSource(std::shared_ptr<fc::DataFilter<float, Dimensionality>> predecessor) :
        m_predecessor(std::move(predecessor)) {
    }

    // PySource interface
public:
    QVector<int> prepare(PreparationProgress &progress) override {
        auto sizes = m_predecessor->prepare(progress);
        QVector<int> result;
        for (size_t i = 0; i < Dimensionality; ++i)
            result.append(sizes[i]);
        return result;
    }

    PyObject *getData(ValidationProgress &progress) override {
        auto container = m_predecessor->getData(progress);
        // TODO: Not implemented yet.
        /* Create an PyObject, owning the container.
         * Create an NumpyArray, pointing to the container data and owning the PyObject.
         * Return the NumpyArray
         * */
    }

    void registerSuccessor(std::weak_ptr<Successor> successor) override {
        m_predecessor->addSuccessor(successor);
    }
    void unregisterSuccessor(Successor *successor) override {
        m_predecessor->removeSuccessor(successor);
    }
};

template <size_t Dimensionality>
std::unique_ptr<PySource> makeSource(const PyFilterNd<Dimensionality> &filter) {
    auto ptr = std::unique_ptr<PySource>(new PyDataFilterSource(filter.getFilter()));
    return ptr;
}

class InvokeTarget {
public:
    virtual ~InvokeTarget() = default;
    virtual QVector<int> prepare(QVector<int> shape) = 0;
    virtual PyObject *getData(PyObject *sources) = 0;
};

class PyInvokeFilter
{
protected:
    std::vector<std::unique_ptr<PySource>> m_sources;
    std::unique_ptr<InvokeTarget> m_invokeTarget;

    QVector<int> invokePrepare(fc::PreparationProgress &progress) {
        // TODO
    }
    PyObject *invokeGetData(fc::ValidationProgress &progress) {
        // TODO
    }

public:
    void addPredecessor(std::unique_ptr<PySource> source) {
        m_sources.push_back(std::move(source));
    }

    void clearPredecessors() {
        m_sources.clear();
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
    PyInvokeFilter();

    void clearPredecessors() {
        m_filter->clearPredecessors();
    }

    void addPredecessor1d(const Filter1d &filter) {
        m_filter->addPredecessor(makeSource(filter));
    }
    void addPredecessor2d(const Filter1d &filter) {
        m_filter->addPredecessor(makeSource(filter));
    }
    void addPredecessor3d(const Filter1d &filter) {
        m_filter->addPredecessor(makeSource(filter));
    }
    void addPredecessor1d(const Filter1d &filter) {
        m_filter->addPredecessor(makeSource(filter));
    }
};

} // namespace pyfc

using InvokeFilter1d = pyfc::PyInvokeFilterNd<1>;
using InvokeFilter2d = pyfc::PyInvokeFilterNd<2>;
using InvokeFilter3d = pyfc::PyInvokeFilterNd<3>;
using InvokeFilter4d = pyfc::PyInvokeFilterNd<4>;

#endif // PYFC_PYINVOKEFILTER_H

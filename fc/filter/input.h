#ifndef FC_FILTER_INPUT_H
#define FC_FILTER_INPUT_H

#include <atomic>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionaltiy>
class Input : public FilterBase, public virtual DataFilter<_ElementType, _Dimensionaltiy>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionaltiy;

private:
	mutable bool m_hasNewData;
	mutable hlp::Threadsafe<Container<ElementType, Dimensionality>> m_newData;
	mutable Container<ElementType, Dimensionality> m_useableData;

public:
	Input()
		: m_hasNewData(false)
	{
	}

	void setData(Container<ElementType, Dimensionality> data)
	{
		auto guard = m_newData.lock();
		if (data.ownsData())
			guard.data() = std::move(data);
		else {
			guard.data() = fc::makeMutableContainer(data.layout().sizes, &guard.data());
#pragma omp parallel
			{
				ndim::copy_omp(data.constData(), guard->mutableData());
			}
		}
		m_hasNewData = true;
	}

private:
	void prepareData() const
	{
		auto guard = m_newData.lock();
		if (!m_hasNewData)
			return;
		m_useableData = std::move(guard.data());
		m_hasNewData = false;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		prepareData();
		progress.throwIfCancelled();
		return m_useableData.layout().sizes;
	}
	virtual Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(progress, recycle);
		return fc::makeConstRefContainer(m_useableData.constData());
	}
};

template <typename ElementType, size_t Dimensionality = 0>
std::shared_ptr<Input<ElementType, Dimensionality>> makeInput()
{
	return std::make_shared<Input<ElementType, Dimensionality>>();
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_INPUT_H

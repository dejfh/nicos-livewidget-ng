#ifndef FC_FILTER_FORWARD_H
#define FC_FILTER_FORWARD_H

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "helper/threadsafe.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionaltiy = 0>
class Forward : public FilterBase, public virtual DataFilter<_ElementType, _Dimensionaltiy>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionaltiy;

	hlp::Threadsafe<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>> m_predecessor;

public:
	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> predecessor() const
	{
		return m_predecessor.get();
	}

	void setPredecessor(std::shared_ptr<const DataFilter<ElementType, Dimensionality>> predecessor)
	{
		if (m_predecessor.unguarded() == predecessor)
			return;
		this->invalidate();
		auto guard = m_predecessor.lock();
		this->unregisterAsSuccessor(guard.data());
		guard->swap(predecessor);
		this->registerAsSuccessor(guard.data());
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		auto predecessor = this->predecessor();
		hlp::throwIfNull(predecessor);
		return predecessor->prepare(progress);
	}

	virtual ndim::Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		auto predecessor = this->predecessor();
		hlp::throwIfNull(predecessor);
		return predecessor->getData(progress, recycle);
	}
};

} // namespace filter
} // namespace fc

#endif // FC_FILTER_FORWARD_H

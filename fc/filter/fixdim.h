#ifndef FC_FILTER_FIXDIM_H
#define FC_FILTER_FIXDIM_H

#include "fc/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionality>
class FixDim : public fc::DataFilter<_ElementType, _Dimensionality>, public fc::FilterBase
{
public:
	using ElementType = _ElementType;
	static const  size_t Dimensionality = _Dimensionality;

private:
	hlp::Threadsafe<std::shared_ptr<const fc::DataFilterVar<ElementType>>> m_predecessor;

public:
	const std::shared_ptr<const fc::DataFilterVar<ElementType>> &predecessor() const
	{
		return m_predecessor.unguarded();
	}
	void setPredecessor(std::shared_ptr<const fc::DataFilterVar<ElementType>> predecessor)
	{
		if (m_predecessor.unguarded() == predecessor)
			return;
		this->invalidate();
		auto guard = m_predecessor.lock();
		this->unregisterAsSuccessor(guard.data());
		std::swap(*guard, predecessor);
		this->registerAsSuccessor(guard.data());
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(fc::PreparationProgress &progress) const override
	{
		auto predecessor = m_predecessor.get();
		hlp::throwIfNull(predecessor);

		ndim::ShapeVar shape = predecessor->prepareVar(progress);
		if (shape.size() != Dimensionality)
			throw std::out_of_range("Unexpected dimensionality.");
		ndim::Sizes<Dimensionality> result;
		std::copy(shape.cbegin(), shape.cend(), result.begin());
		return result;
	}
	virtual ndim::Container<ElementType, Dimensionality> getData(
			fc::ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		auto predecessor = m_predecessor.get();
		hlp::throwIfNull(predecessor);

		ndim::ContainerVar<ElementType> data = predecessor->getDataVar(progress, recycle);
		return data.template fixDimensionality<Dimensionality>();
	}
};

} // namespace filter
} // namespace fc

#endif // FC_FILTER_FIXDIM_H

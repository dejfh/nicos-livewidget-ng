#ifndef FC_FILTER_PILE_H
#define FC_FILTER_PILE_H

#include <QVector>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "helper/qt/vector.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _PredecessorDimensionality>
class Pile : public FilterBase, public virtual DataFilter<_ElementType, _PredecessorDimensionality + 1>
{
public:
	using ElementType = _ElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = PredecessorDimensionality + 1;

private:
	size_t m_insertDimension;
	hlp::Threadsafe<QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>>> m_predecessors;
	mutable ndim::Sizes<PredecessorDimensionality> m_predecessorSizes;

public:
	Pile(size_t insertDimension = PredecessorDimensionality)
		: m_insertDimension(insertDimension)
	{
	}

	size_t insertDimension() const
	{
		return m_insertDimension;
	}
	void setInsertDimension(size_t insertDimension)
	{
		if (m_insertDimension == insertDimension)
			return;
		this->invalidate();
		m_insertDimension = insertDimension;
	}

	QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>> predecessors() const
	{
		return m_predecessors.get();
	}
	void setPredecessors(QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>> predecessors)
	{
		this->invalidate();
		auto guard = m_predecessors.lock();
		this->unregisterAsSuccessor(guard.data());
		predecessors.swap(guard.data()); // Order of destruction ensures that teferences to old predecessors are released after the lock has been released.
		this->registerAsSuccessor(guard.data());
	}

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const override
	{
		auto predecessors = this->predecessors();
		size_t insertDimension = this->m_insertDimension;
		progress.throwIfCancelled();
		ndim::Sizes<PredecessorDimensionality> sizes;
		bool first = true;
		for (const auto &predecessor : predecessors) {
			if (first)
				sizes = hlp::throwIfNull(predecessor)->prepare(progress);
			else if (sizes != hlp::throwIfNull(predecessor)->prepare(progress))
				throw ::std::out_of_range("Pile predecessors have different sizes.");
		}
		m_predecessorSizes = sizes;
		return hlp::array::insert(sizes, insertDimension, size_t(predecessors.size()));
	}

	virtual ndim::Container<ElementType, ResultDimensionality> getData(
		ValidationProgress &progress, ndim::Container<ElementType, ResultDimensionality> *recycle) const override
	{
		auto predecessors = this->predecessors();
		size_t insertDimension = this->m_insertDimension;
		progress.throwIfCancelled();

		ndim::Container<ElementType, ResultDimensionality> result =
			ndim::makeMutableContainer(hlp::array::insert(m_predecessorSizes, insertDimension, size_t(predecessors.size())), recycle);
		ndim::pointer<ElementType, ResultDimensionality> data = result.mutableData();

		size_t index = 0;
		for (auto it = predecessors.constBegin(), end = predecessors.constEnd(); it != end; ++it, ++index) {
			const std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> &predecessor = *it;
			hlp::throwIfNull(predecessor);
			fc::getData(progress, predecessor, data.removeDimension(insertDimension, index));
		}
		return result;
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<Pile<ElementType, PredecessorDimensionality>> makePile(
	const QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>> &predecessors,
	size_t insertDimension = PredecessorDimensionality)
{
	auto filter = std::make_shared<Pile<ElementType, PredecessorDimensionality>>(insertDimension);
	filter->setPredecessors(predecessors);
	return filter;
}

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<Pile<ElementType, PredecessorDimensionality>> makePile(size_t insertDimension = PredecessorDimensionality)
{
	return makePile(QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>>(), insertDimension);
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_PILE_H

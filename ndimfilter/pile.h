#ifndef NDIMFILTER_PILE_H
#define NDIMFILTER_PILE_H

#include "filter/filterbase.h"
#include "ndimfilter/filter.h"
#include <QList>

#include "safecast.h"
#include "helper/threadsafe.h"

#include "helper/helper.h"
#include "filter/gethelper.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _PredecessorDimensionality>
class pile : public PredecessorVectorFilterBase<DataFilter<_ElementType, _PredecessorDimensionality>>,
			 public virtual NoConstDataFilter<_ElementType, _PredecessorDimensionality + 1>
{
public:
	using ElementType = _ElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = PredecessorDimensionality + 1;

private:
	size_t m_insertDimension;
	mutable ndim::sizes<PredecessorDimensionality> m_predecessorSizes;

public:
	pile(size_t insertDimension = PredecessorDimensionality)
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

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessors = this->predecessors();
		size_t insertDimension = m_insertDimension;
		progress.throwIfCancelled();

		if (predecessors.empty()) {
			return ndim::sizes<ResultDimensionality>();
		}

		bool first = true;
		ndim::sizes<PredecessorDimensionality> sizes;
		for (const std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> &predecessor : predecessors) {
			hlp::notNull(predecessor);
			if (first)
				sizes = predecessor->prepare(progress);
			else if (sizes != predecessor->prepare(progress))
				throw ::std::out_of_range("Pile predecessors have different sizes.");
		}
		m_predecessorSizes = sizes;
		return sizes.addDimension(insertDimension, predecessors.size());
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, ResultDimensionality> &result,
		OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessors = this->predecessors();
		size_t insertDimension = m_insertDimension;
		progress.throwIfCancelled();

		result.resize(m_predecessorSizes.addDimension(insertDimension, predecessors.size()));
		ndim::pointer<ElementType, ResultDimensionality> data = result.pointer();

		size_t index = 0;
		for (auto it = predecessors.cbegin(), end = predecessors.cend(); it != end; ++it, ++index) {
			const std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> &predecessor = *it;
			hlp::notNull(predecessor);
			filter::getData(progress, predecessor, data.removeDimension(insertDimension, index));
		}
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<pile<ElementType, PredecessorDimensionality>> makePile(
	const QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>> &predecessors,
	size_t insertDimension = PredecessorDimensionality)
{
	auto filter = std::make_shared<pile<ElementType, PredecessorDimensionality>>(insertDimension);
	filter->setPredecessors(predecessors);
	return filter;
}

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<pile<ElementType, PredecessorDimensionality>> makePile(size_t insertDimension = PredecessorDimensionality)
{
	return makePile(QVector<std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>>>(), insertDimension);
}

} // namespace ndimfilter

#endif // NDIMFILTER_PILE_H

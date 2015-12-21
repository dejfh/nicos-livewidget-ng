#ifndef NDIMFILTER_EXTEND_H
#define NDIMFILTER_EXTEND_H

#include <initializer_list>
#include <array>

#include "ndim/pointer.h"
#include "ndimfilter/filter.h"
#include "filter/gethelper.h"
#include "filter/typetraits.h"

#include "ndim/algorithm_omp.h"

#include "helper/helper.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class extend : public SinglePredecessorFilterBase<DataFilter<_ElementType, _PredecessorDimensionality>>,
			   public DataFilter<_ElementType, _ResultDimensionality>
{
public:
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t ExtendDimensionality = ResultDimensionality - PredecessorDimensionality;
	static_assert(PredecessorDimensionality <= ResultDimensionality, "Can not extend to lower dimensionality.");

	using ElementType = _ElementType;

private:
	std::array<size_t, ExtendDimensionality> m_extendDimensions;
	ndim::sizes<ExtendDimensionality> m_extendSizes;

public:
	extend(std::array<size_t, ExtendDimensionality> extendDimensions, ndim::sizes<ExtendDimensionality> extendSizes)
		: m_extendDimensions(extendDimensions)
		, m_extendSizes(extendSizes)
	{
	}

	std::array<size_t, ExtendDimensionality> extendDimensions() const
	{
		return m_extendDimensions;
	}
	void setExtendDimensions(std::array<size_t, ExtendDimensionality> extendDimensions)
	{
		if (m_extendDimensions == extendDimensions)
			return;
		this->invalidate();
		m_extendDimensions = extendDimensions;
	}

	ndim::sizes<ExtendDimensionality> extendSizes() const
	{
		return m_extendSizes;
	}
	void setExtendSizes(ndim::sizes<ExtendDimensionality> extendSizes)
	{
		if (m_extendSizes == extendSizes)
			return;
		this->invalidate();
		m_extendSizes = extendSizes;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto extendDimensions = m_extendDimensions;
		auto extendSizes = m_extendSizes;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		ndim::sizes<PredecessorDimensionality> predecessorSizes = predecessor->prepareConst(progress);
		return predecessorSizes.addDimensions(extendSizes, extendDimensions);
	}

	virtual ndim::sizes<ResultDimensionality> prepareConst(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto extendDimensions = m_extendDimensions;
		auto extendSizes = m_extendSizes;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		ndim::sizes<PredecessorDimensionality> predecessorSizes = predecessor->prepareConst(progress);
		return predecessorSizes.addDimensions(extendSizes, extendDimensions);
	}

	virtual void getData(ValidationProgress &progress, Container<ElementType, ResultDimensionality> &result,
		OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto extendDimensions = m_extendDimensions;
		auto extendSizes = m_extendSizes;
		progress.throwIfCancelled();

		Container<const ElementType, PredecessorDimensionality> container = filter::getConstData(progress, predecessor);
		ndim::pointer<const ElementType, ResultDimensionality> source = container.m_pointer.addDimensions(extendDimensions, extendSizes);

		result.resize(source.sizes);
		ndim::copy_omp(source, result.pointer());
	}
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, ResultDimensionality> &result,
		OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto extendDimensions = m_extendDimensions;
		auto extendSizes = m_extendSizes;
		progress.throwIfCancelled();

		Container<const ElementType, PredecessorDimensionality> container = filter::getConstData(progress, predecessor);

		result.m_buffer = std::move(container.m_buffer);
		result.m_pointer = container.m_pointer.addDimensions(extendDimensions, extendSizes);
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _Predecessor, size_t _ExtendDimensionality>
std::shared_ptr<
	extend<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value, DimensionalityOf_t<_Predecessor>::value + _ExtendDimensionality>>
makeExtend(std::shared_ptr<_Predecessor> predecessor, std::array<size_t, _ExtendDimensionality> extendDimensions,
	std::array<size_t, _ExtendDimensionality> extendSizes)
{
	auto filter = std::make_shared<extend<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value,
		DimensionalityOf_t<_Predecessor>::value + _ExtendDimensionality>>(extendDimensions, extendSizes);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace ndimfilter

#endif // NDIMFILTER_EXTEND_H

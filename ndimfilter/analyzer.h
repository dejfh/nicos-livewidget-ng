#ifndef NDIMFILTER_ANALYZER_H
#define NDIMFILTER_ANALYZER_H

#include <numeric>

#include "helper/helper.h"

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/gethelper.h"
#include "filter/typetraits.h"

#include "ndim/pointer.h"
#include "ndim/buffer.h"

#include "ndimfilter/filter.h"

#include "ndimdata/analyzer.h"
#include "ndimdata/statistic.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _Dimensionality>
class Analyzer : public SinglePredecessorFilterBase<DataFilter<_ElementType, _Dimensionality>>,
				 public virtual NoConstDataFilter<ndimdata::DataStatistic>
{
	using PredecessorElementType = _ElementType;
	static const size_t PredecessorDimensionality = _Dimensionality;

	double m_roiRatio;
	double m_displayRatio;
	size_t m_binCount;

	const QString m_description;

public:
	Analyzer(QString description)
		: m_roiRatio(.95)
		, m_displayRatio(.5)
		, m_binCount(256)
		, m_description(std::move(description))
	{
	}

	// DataFilter interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<ndimdata::DataStatistic>>) const override
	{
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		auto sizes = predecessor->prepare(progress);

		progress.addStep(ndimdata::analyzeDuration(sizes.size()), m_description);

		return ndim::sizes<0>();
	}
	virtual void getData(ValidationProgress &progress, Container<ndimdata::DataStatistic, 0> &result,
		OverloadDummy<DataFilter<ndimdata::DataStatistic>>) const override
	{
		auto predecessor = this->predecessor();
		double roiRatio = m_roiRatio;
		double displayRatio = m_displayRatio;
		size_t binCount = m_binCount;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		Container<PredecessorElementType, PredecessorDimensionality> container;
		predecessor->getData(progress, container);
		auto pointer = container.pointer();

		if (pointer.size() == 0)
			throw std::runtime_error("can not analyze region of size 0.");

		// TODO: ndimdata::analyze needs contiguous data. Change ndimdata::analyze, or change constraints of Predecessor::getData.

		ndim::Buffer<PredecessorElementType, PredecessorDimensionality> contiguousBuffer;
		if (!pointer.isContiguous()) {
			contiguousBuffer.resize(pointer.sizes);
#pragma omp parallel
			{
				ndim::copy_omp(pointer, contiguousBuffer.pointer());
			}
			pointer = contiguousBuffer.pointer();
		}

		result.resize();

		ndimdata::analyze(pointer.data, pointer.size(), *result.m_pointer.data, roiRatio, displayRatio, binCount, progress);
		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _Predecessor>
std::shared_ptr<Analyzer<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>> makeAnalyzer(
	std::shared_ptr<_Predecessor> predecessor, QString description)
{
	auto filter = std::make_shared<Analyzer<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>>(std::move(description));
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter

#endif // NDIMFILTER_ANALYZER_H

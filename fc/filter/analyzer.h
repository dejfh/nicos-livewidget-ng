#ifndef FC_FILTER_ANALYZER_H
#define FC_FILTER_ANALYZER_H

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "ndimdata/analyzer.h"
#include "ndimdata/statistic.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionality>
class AnalyzerHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, _Dimensionality>>
{
public:
	using PredecessorElementType = _ElementType;
	static const size_t PredecessorDimensionality = _Dimensionality;

	double roiRatio;
	double displayRatio;
	size_t binCount;

	QString description;

	AnalyzerHandler(const QString &description, double roiRatio, double displayRatio, size_t binCount)
		: roiRatio(roiRatio)
		, displayRatio(displayRatio)
		, binCount(binCount)
		, description(description)
	{
	}

	ndim::sizes<0> prepare(PreparationProgress &progress) const
	{
		auto sizes = std::get<0>(this->preparePredecessors(progress));

		progress.addStep(ndimdata::analyzeDuration(sizes.size()), description);

		return ndim::Sizes<0>();
	}
	ndim::Container<ndimdata::DataStatistic, 0> getData(ValidationProgress &progress, ndim::Container<ndimdata::DataStatistic, 0> *recycle) const
	{
		auto input = std::get<0>(this->getPredecessorsData(progress));

		if (input.layout().size() == 0)
			throw std::runtime_error("can not analyze region of size 0.");

		// TODO: ndimdata::analyze needs contiguous data. Change ndimdata::analyze.

		ndim::Container<PredecessorElementType, PredecessorDimensionality> contiguousBuffer;
		ndim::pointer<PredecessorElementType, PredecessorDimensionality> mutableData;

		mutableData = input.mutableData();
		if (!mutableData || !mutableData.isContiguous()) {
			auto constData = input.constData();
			contiguousBuffer.resize(constData.sizes);
			mutableData = contiguousBuffer.mutableData();
#pragma omp parallel
			{
				ndim::copy_omp(constData, mutableData);
			}
		}

		auto result = ndim::makeMutableContainer(recycle);
		ndimdata::DataStatistic &stat = result.mutableData().first();

		ndimdata::analyze(mutableData.data, mutableData.size(), stat, roiRatio, displayRatio, binCount, progress);
		progress.advanceStep();

		return result;
	}
};

template <typename ElementType, size_t Dimensionality>
class Analyzer : public HandlerDataFilterWithDescriptionBase<ndimdata::DataStatistic, 0, AnalyzerHandler<ElementType, Dimensionality>>
{
public:
	Analyzer(const QString &description, double roiRatio, double displayRatio, size_t binCount)
		: HandlerDataFilterWithDescriptionBase<ndimdata::DataStatistic, 0, AnalyzerHandler<ElementType, Dimensionality>>(
			  description, roiRatio, displayRatio, binCount)
	{
	}

	double roiRatio() const
	{
		return this->m_handler.unguarded().roiRatio;
	}
	void setRoiRatio(double roiRatio)
	{
		if (this->m_handler.unguarded().roiRatio == roiRatio)
			return;
		this->invalidate();
		this->m_handler.lock()->roiRatio = roiRatio;
	}

	double displayRatio() const
	{
		return this->m_handler.unguarded().displayRatio;
	}
	void setDisplayRatio(double displayRatio)
	{
		if (this->m_handler.unguarded().displayRatio == displayRatio)
			return;
		this->invalidate();
		this->m_handler.lock()->displayRatio = displayRatio;
	}

	size_t binCount() const
	{
		return this->m_handler.unguarded().binCount;
	}
	void setBinCount(size_t binCount)
	{
		if (this->m_handler.unguarded().binCount == binCount)
			return;
		this->invalidate();
		this->m_handler.lock()->binCount = binCount;
	}
};

template <typename _Predecessor>
std::shared_ptr<Analyzer<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>> makeAnalyzer(
	std::shared_ptr<_Predecessor> predecessor, QString description)
{
	auto filter = std::make_shared<Analyzer<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>>(description, .95, .5, 1024);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_ANALYZER_H

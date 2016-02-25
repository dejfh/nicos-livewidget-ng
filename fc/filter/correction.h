#ifndef CORRECTION
#define CORRECTION

#include "fc/datafilterbase.h"

#include "helper/helper.h"

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionality>
class CorrectionHandler : public DataFilterHandlerBase<DataFilter<float, _Dimensionality>, DataFilter<_ElementType, _Dimensionality>,
							  DataFilter<_ElementType, _Dimensionality>>
{
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

	using PredecessorElementType = ElementType;
	static const size_t PredecessorDimensionality = Dimensionality;

	using ResultElementType = float;
	static const size_t ResultDimensionality = Dimensionality;

public:
	const std::shared_ptr<const DataFilter<PredecessorElementType, PredecessorDimensionality>> &darkImage() const
	{
		return std::get<1>(this->predecessors);
	}
	const std::shared_ptr<const DataFilter<PredecessorElementType, PredecessorDimensionality>> &openBeam() const
	{
		return std::get<2>(this->predecessors);
	}
	void setDarkIamge(std::shared_ptr<const DataFilter<PredecessorElementType, PredecessorDimensionality>> darkImage)
	{
		std::get<1>(this->predecessors) = std::move(darkImage);
	}
	void setOpenBeam(std::shared_ptr<const DataFilter<PredecessorElementType, PredecessorDimensionality>> openBeam)
	{
		std::get<2>(this->predecessors) = std::move(openBeam);
	}

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		ndim::Sizes<Dimensionality> sizes = hlp::notNull(this->predecessor())->prepare(progress);
		if (this->darkImage())
			this->darkImage()->prepare(progress);
		if (this->openBeam())
			this->openBeam()->prepare(progress);
		return sizes;
	}

	virtual Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		Container<PredecessorElementType, PredecessorDimensionality> *predecessorRecycle = nullptr;
		hlp::assignIfAssignable(predecessorRecycle, recycle);

		Container<ResultElementType, ResultDimensionality> data = this->predecessor()->getData(progress, recycle);
		ndim::pointer<const ResultElementType, ResultDimensionality> data_ptr = data.constData();
		bool hasDarkImage = this->darkImage();
		bool hasOpenBeam = this->openBeam();
		Container<ResultElementType, ResultDimensionality> result;
		if (data.isMutable())
			result = std::move(data);
		else
			result = fc::makeMutableContainer(data_ptr.sizes, recycle);
		ndim::pointer<ResultElementType, ResultDimensionality> result_ptr = result.mutableData();

		if (hasDarkImage && hasOpenBeam) {
			auto op = [](float v, float d, float o) { return (v - d) / (o - d); };
			Container<PredecessorElementType, PredecessorDimensionality> darkImage = this->darkImage()->getData(progress, predecessorRecycle);
			Container<PredecessorElementType, PredecessorDimensionality> openBeam = this->openBeam()->getData(progress, predecessorRecycle);
#pragma omp parallel
			{
				ndim::transform_omp(result_ptr, op, data_ptr, darkImage.constData(), openBeam.constData());
			}
		} else if (hasDarkImage) {
			auto op = [](float v, float d) { return v - d; };
			Container<PredecessorElementType, PredecessorDimensionality> darkImage = this->darkImage()->getData(progress, predecessorRecycle);
#pragma omp parallel
			{
				ndim::transform_omp(result_ptr, op, data_ptr, darkImage.constData());
			}
		} else if (hasOpenBeam) {
			auto op = [](float v, float o) { return v / o; };
			Container<PredecessorElementType, PredecessorDimensionality> openBeam = this->openBeam()->getData(progress, predecessorRecycle);
#pragma omp parallel
			{
				ndim::transform_omp(result_ptr, op, data_ptr, openBeam.constData());
			}
		} else if (data_ptr.data != result_ptr.data) {
#pragma omp parallel
			{
				ndim::copy_omp(data_ptr, result_ptr);
			}
		}
		return result;
	}
};

template <typename ElementType, size_t Dimensionality>
class Correction : public HandlerDataFilterBase<float, Dimensionality, CorrectionHandler<ElementType, Dimensionality>>
{
public:
	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> darkImage() const
	{
		return this->template predecessor<1>();
	}
	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> openBeam() const
	{
		return this->template predecessor<2>();
	}
	void setDarkIamge(std::shared_ptr<const DataFilter<ElementType, Dimensionality>> darkImage)
	{
		this->template setPredecessor<1>(darkImage);
	}
	void setOpenBeam(std::shared_ptr<const DataFilter<ElementType, Dimensionality>> openBeam)
	{
		this->template setPredecessor<2>(openBeam);
	}
};

} // namespace filter
} // namespace fc

#endif // CORRECTION

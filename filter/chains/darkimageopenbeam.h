#ifndef FILTER_CHAINS_DARKIMAGEOPENBEAM_H
#define FILTER_CHAINS_DARKIMAGEOPENBEAM_H

#include <memory>

#include "filter/filter.h"

#include "filter/buffer.h"
#include "ndimfilter/mean.h"
#include "ndimfilter/transform.h"

#include "filter/chains/fitspile.h"

namespace filter
{
namespace chains
{

template <typename _ElementType, size_t _Dimensionality>
class DarkImageAndOpenBeamChain
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	FitsPileChain<ElementType, Dimensionality> m_darkImagePile;
	FitsPileChain<ElementType, Dimensionality> m_openBeamPile;

	std::shared_ptr<filter::Buffer<ElementType, Dimensionality>> m_darkImageBuffer;
	std::shared_ptr<filter::Buffer<ElementType, Dimensionality>> m_openBeamBuffer;

public:
	DarkImageAndOpenBeamChain()
	{
		// Calcualte mean values
		auto darkImageMean = filter::makeMean("Generating dark image...", m_darkImagePile.pileBuffer(), 0);
		// Buffer resulting dark images for repeated use
		m_darkImageBuffer = filter::makeBuffer(darkImageMean);

		// Calculate mean values
		auto openBeamMean = filter::makeMean("Generating open beam...", m_openBeamPile.pileBuffer(), 0);
		// Subtract dark image
		auto openBeamSubstractDarkImage =
			filter::makeTransform("Subtract dark image from open beam...", std::minus<float>(), 1, openBeamMean, m_darkImageBuffer);
		// Buffer resulting open beam image for repeated use
		m_openBeamBuffer = filter::makeBuffer(openBeamSubstractDarkImage);
	}

	void setDarkImages(const QStringList &filenames)
	{
		m_darkImagePile.setFilenames(filenames);
	}
	void setOpenBeam(const QStringList &filenames)
	{
		m_openBeamPile.setFilenames(filenames);
	}

	std::shared_ptr<const filter::Buffer<ElementType, Dimensionality>> darkImageBuffer() const
	{
		return m_darkImageBuffer;
	}
	std::shared_ptr<const filter::Buffer<ElementType, Dimensionality>> openBeamBuffer() const
	{
		return m_openBeamBuffer;
	}
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_DARKIMAGEOPENBEAM_H

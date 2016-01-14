#ifndef FILTER_CHAINS_DARKIMAGEOPENBEAM_H
#define FILTER_CHAINS_DARKIMAGEOPENBEAM_H

#include <memory>

#include "filter/filter.h"

#include "filter/buffer.h"
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

	using SharedFilterPtr = std::shared_ptr<DataFilter<ElementType, Dimensionality>>;

private:
	FitsPileChain<ElementType, Dimensionality> m_darkImagePile;
	FitsPileChain<ElementType, Dimensionality> m_openBeamPile;

	std::shared_ptr<filter::Buffer<ElementType, Dimensionality>> m_darkImageBuffer;
	std::shared_ptr<filter::Buffer<ElementType, Dimensionality>> m_openBeamBuffer;

public:
	DarkImageAndOpenBeamChain();

	void setDarkImages(const QStringList &filenames);
	void setOpenBeam(const QStringList &filenames);

	std::shared_ptr<const filter::Buffer<_ElementType, _Dimensionality>> darkImageBuffer() const;
	std::shared_ptr<const filter::Buffer<_ElementType, _Dimensionality>> openBeamBuffer() const;
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_DARKIMAGEOPENBEAM_H

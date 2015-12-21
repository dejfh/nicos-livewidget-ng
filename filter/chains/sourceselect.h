#ifndef HELPER_CHAINS_SOURCESELECT_H
#define HELPER_CHAINS_SOURCESELECT_H

#include <memory>

#include "filter/filter.h"
#include "filter/buffer.h"
#include "filter/switch.h"

namespace filter
{
namespace chains
{

template <typename _ElementType, size_t _Dimensionality>
class SourceSelectChain
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	std::shared_ptr<filter::Switch<ElementType, Dimensionality>> m_normalizeSwitch;
	std::shared_ptr<filter::Switch<ElementType, Dimensionality>> m_sourceSwitch;

public:
	SourceSelectChain(std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality>> darkImage,
		std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality>> openBeam,
		std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality>> source)
	{
		// Subtract dark image
		auto imageSubstractDarkImage = filter::makeTransform("Subtracting dark image...", std::minus<float>(), 1, source, darkImage);
		// Buffer image for repeated use
		auto imageBuffer = filter::makeBuffer(imageSubstractDarkImage);

		// Normalize with open beam image
		auto imageNormalize = filter::makeTransform("Normalizing image...", std::divides<float>(), 1, imageBuffer, openBeam);
		// Make Normalization skippable
		m_normalizeSwitch = filter::makeSwitch(imageBuffer, imageNormalize);

		m_sourceSwitch = filter::makeSwitch(m_normalizeSwitch, darkImage, openBeam);
	}

	void setNormalize(bool normalize)
	{
		m_normalizeSwitch->setSelection(normalize ? 1 : 0);
	}
	void setSource(int source)
	{
		m_sourceSwitch->setSelection(source);
	}

	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> sourceFilter() const
	{
		return m_sourceSwitch;
	}
};

} // namespace chains
} // namespace filter

#endif // HELPER_CHAINS_SOURCESELECT_H

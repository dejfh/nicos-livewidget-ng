#ifndef HELPER_CHAINS_SOURCESELECT_H
#define HELPER_CHAINS_SOURCESELECT_H

#include <memory>

#include "fc/filter.h"
#include "fc/buffer.h"
#include "fc/switch.h"

namespace fc
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
	std::shared_ptr<fc::Switch<ElementType, Dimensionality>> m_normalizeSwitch;
	std::shared_ptr<fc::Switch<ElementType, Dimensionality>> m_sourceSwitch;

public:
	SourceSelectChain(std::shared_ptr<const fc::DataFilter<ElementType, Dimensionality>> darkImage,
		std::shared_ptr<const fc::DataFilter<ElementType, Dimensionality>> openBeam,
		std::shared_ptr<const fc::DataFilter<ElementType, Dimensionality>> source)
	{
		// Subtract dark image
		auto imageSubstractDarkImage = fc::makeTransform("Subtracting dark image...", std::minus<float>(), 1, source, darkImage);
		// Buffer image for repeated use
		auto imageBuffer = fc::makeBuffer(imageSubstractDarkImage);

		// Normalize with open beam image
		auto imageNormalize = fc::makeTransform("Normalizing image...", std::divides<float>(), 1, imageBuffer, openBeam);
		// Make Normalization skippable
		m_normalizeSwitch = fc::makeSwitch(imageBuffer, imageNormalize);

		m_sourceSwitch = fc::makeSwitch(m_normalizeSwitch, darkImage, openBeam);
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
} // namespace fc

#endif // HELPER_CHAINS_SOURCESELECT_H

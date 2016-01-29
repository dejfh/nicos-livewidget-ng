#include "fc/chains/darkimageopenbeam.h"

#include "ndimfilter/mean.h"

template <typename ElementType, size_t Dimensionality>
fc::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::DarkImageAndOpenBeamChain()
{
	// Calcualte mean values
	SharedFilterPtr darkImageMean = fc::makeMean("Generating dark image...", m_darkImagePile.pileBuffer(), 0);
	// Buffer resulting dark images for repeated use
	m_darkImageBuffer = fc::makeBuffer(darkImageMean);

	// Calculate mean values
	SharedFilterPtr openBeamMean = fc::makeMean("Generating open beam...", m_openBeamPile.pileBuffer(), 0);
	// Subtract dark image
	SharedFilterPtr openBeamSubstractDarkImage =
		fc::makeTransform("Subtract dark image from open beam...", std::minus<float>(), openBeamMean, m_darkImageBuffer);

	// Buffer resulting open beam image for repeated use
	m_openBeamBuffer = fc::makeBuffer(openBeamSubstractDarkImage);
}

template <typename ElementType, size_t Dimensionality>
void fc::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::setDarkImages(const QStringList &filenames)
{
	m_darkImagePile.setFilenames(filenames);
}

template <typename ElementType, size_t Dimensionality>
void fc::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::setOpenBeam(const QStringList &filenames)
{
	m_openBeamPile.setFilenames(filenames);
}

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<const fc::Buffer<ElementType, Dimensionality>>
fc::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::darkImageBuffer() const
{
	return m_darkImageBuffer;
}

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<const fc::Buffer<ElementType, Dimensionality>>
fc::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::openBeamBuffer() const
{
	return m_openBeamBuffer;
}

template class fc::chains::DarkImageAndOpenBeamChain<float, 2>;

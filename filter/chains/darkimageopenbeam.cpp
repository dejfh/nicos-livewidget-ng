#include "filter/chains/darkimageopenbeam.h"

#include "ndimfilter/mean.h"

template <typename ElementType, size_t Dimensionality>
filter::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::DarkImageAndOpenBeamChain()
{
	// Calcualte mean values
	SharedFilterPtr darkImageMean = filter::makeMean("Generating dark image...", m_darkImagePile.pileBuffer(), 0);
	// Buffer resulting dark images for repeated use
	m_darkImageBuffer = filter::makeBuffer(darkImageMean);

	// Calculate mean values
	SharedFilterPtr openBeamMean = filter::makeMean("Generating open beam...", m_openBeamPile.pileBuffer(), 0);
	// Subtract dark image
	SharedFilterPtr openBeamSubstractDarkImage =
		filter::makeTransform("Subtract dark image from open beam...", std::minus<float>(), openBeamMean, m_darkImageBuffer);

	// Buffer resulting open beam image for repeated use
	m_openBeamBuffer = filter::makeBuffer(openBeamSubstractDarkImage);
}

template <typename ElementType, size_t Dimensionality>
void filter::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::setDarkImages(const QStringList &filenames)
{
	m_darkImagePile.setFilenames(filenames);
}

template <typename ElementType, size_t Dimensionality>
void filter::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::setOpenBeam(const QStringList &filenames)
{
	m_openBeamPile.setFilenames(filenames);
}

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<const filter::Buffer<ElementType, Dimensionality>>
filter::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::darkImageBuffer() const
{
	return m_darkImageBuffer;
}

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<const filter::Buffer<ElementType, Dimensionality>>
filter::chains::DarkImageAndOpenBeamChain<ElementType, Dimensionality>::openBeamBuffer() const
{
	return m_openBeamBuffer;
}

template class filter::chains::DarkImageAndOpenBeamChain<float, 2>;

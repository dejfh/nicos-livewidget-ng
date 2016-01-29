#ifndef FILTER_CHAINS_DATAPROCESS_H
#define FILTER_CHAINS_DATAPROCESS_H

#include <memory>

#include "fc/filter.h"
#include "fc/buffer.h"
#include "fc/switch.h"

namespace fc
{
namespace chains
{

template <typename _ElementType, size_t _Dimensionality>
class DataProcessChain
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	std::shared_ptr<fc::Buffer<ElementType, Dimensionality>> m_processedBuffer;

public:
	DataProcessChain(std::shared_ptr<const fc::DataFilter<ElementType, Dimensionality>> source)
	{
		m_processedBuffer = fc::makeBuffer(source);
	}

	std::shared_ptr<const fc::Buffer<ElementType, Dimensionality>> processedBuffer() const
	{
		return m_processedBuffer;
	}
};

} // namespace chains
} // namespace fc

#endif // FILTER_CHAINS_DATAPROCESS_H

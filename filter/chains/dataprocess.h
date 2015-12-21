#ifndef FILTER_CHAINS_DATAPROCESS_H
#define FILTER_CHAINS_DATAPROCESS_H

#include <memory>

#include "filter/filter.h"
#include "filter/buffer.h"
#include "filter/switch.h"

namespace filter
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
	std::shared_ptr<filter::Buffer<ElementType, Dimensionality>> m_processedBuffer;

public:
	DataProcessChain(std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality>> source)
	{
		m_processedBuffer = filter::makeBuffer(source);
	}

	std::shared_ptr<const filter::Buffer<ElementType, Dimensionality>> processedBuffer() const
	{
		return m_processedBuffer;
	}
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_DATAPROCESS_H

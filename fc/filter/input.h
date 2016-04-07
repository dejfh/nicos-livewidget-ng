#ifndef FC_FILTER_INPUT_H
#define FC_FILTER_INPUT_H

#include <atomic>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _Dimensionaltiy>
class Input : public FilterBase, public virtual DataFilter<_ElementType, _Dimensionaltiy>, public virtual Validatable
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionaltiy;

private:
	mutable hlp::Threadsafe<ndim::Container<ElementType, Dimensionality>> m_newData;
	mutable ndim::Container<ElementType, Dimensionality> m_useableData;

public:
	Input()
	{
	}

	void setData(ndim::Container<ElementType, Dimensionality> data)
	{
		this->invalidate();
		auto guard = m_newData.lock();
		if (data.ownsData())
			guard.data() = std::move(data);
		else {
			guard.data() = ndim::makeMutableContainer(data.layout().sizes, &guard.data());
#pragma omp parallel
			{
				ndim::copy_omp(data.constData(), guard->mutableData());
			}
		}
	}

private:
	void prepareData() const
	{
		auto guard = m_newData.lock();
		if (!guard.data().hasData())
			return;
		m_useableData = std::move(guard.data());
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		progress.appendValidatable(std::shared_ptr<const Validatable>(this->shared_from_this(), this));
		prepareData();
		progress.throwIfCancelled();
		return m_useableData.layout().sizes;
	}
	virtual ndim::Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(progress, recycle);
		return m_useableData.constData();
	}

	// Validatable interface
public:
	virtual void prepareValidation(PreparationProgress &progress) const override
	{
		this->prepare(progress);
	}
	virtual void validate(ValidationProgress &progress) const override
	{
		this->getData(progress, nullptr);
	}
	virtual bool isValid() const override
	{
		return !m_newData.lockConst().data().hasData();
	}
};

template <typename ElementType, size_t Dimensionality = 0>
std::shared_ptr<Input<ElementType, Dimensionality>> makeInput()
{
	return std::make_shared<Input<ElementType, Dimensionality>>();
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_INPUT_H

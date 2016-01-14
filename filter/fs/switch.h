#ifndef FILTER_FS_SWITCH_H
#define FILTER_FS_SWITCH_H

#include <QVector>

#include "filter/datafilter.h"
#include "filter/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include "helper/qt/vector.h"

namespace filter
{
namespace fs
{

struct SwitchControl {
	virtual int selection() const = 0;
	virtual void setSelection(int selection) = 0;
};

template <typename _ElementType, size_t _Dimensionality = 0>
class Switch : public FilterBase, public virtual SwitchControl, public virtual DataFilter<_ElementType, _Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	int m_selection;
	hlp::Threadsafe<QVector<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>>> m_predecessors;

protected:
	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> selectedPredecessor() const
	{
		auto guard = this->m_predecessors.lockConst();
		int selection = m_selection;
		return guard.data().value(selection);
	}

public:
	Switch(int selection = 0)
		: m_selection(selection)
	{
	}

	QVector<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>> predecessors() const
	{
		return m_predecessors.get();
	}
	void setPredecessors(const QVector<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>> &predecessors)
	{
		this->invalidate();
		auto guard = m_predecessors.lock();
		this->unregisterSuccessor(guard.data());
		guard.data() = std::move(predecessors);
		this->registerSuccessor(guard.data());
	}

	// SwitchControl interface
public:
	virtual int selection() const override
	{
		return m_selection;
	}
	virtual void setSelection(int selection) override
	{
		if (m_selection == selection)
			return;
		this->invalidate();
		m_selection = selection;
	}

	// Invalidatable interface
public:
	virtual void predecessorInvalidated(const Predecessor *predecessor) override
	{
		int selection = m_selection;
		{
			auto guard = this->m_predecessors.lock();
			if (selection < 0 || selection >= guard.data().size())
				return;
			if (guard.data().at(selection).get() != predecessor)
				return;
		}
		this->invalidate();
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		return predecessor->prepare(progress);
	}
	virtual Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, Container<ElementType, Dimensionality> *recycle) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		return predecessor->getData(progress, recycle);
	}
};

template <typename _ElementType, size_t _Dimensionality>
std::shared_ptr<Switch<_ElementType, _Dimensionality>> makeSwitch()
{
	return std::make_shared<Switch<_ElementType, _Dimensionality>>();
}

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<Switch<ElementType, Dimensionality>> makeSwitch(QVector<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>> predecessors)
{
	auto filter = std::make_shared<Switch<ElementType, Dimensionality>>();
	filter->setPredecessors(predecessors);
	return filter;
}

template <typename... _PredecessorTypes>
std::shared_ptr<Switch<ElementTypeOf_t<_PredecessorTypes...>, DimensionalityOf_t<_PredecessorTypes...>::value>> makeSwitch(
	std::shared_ptr<_PredecessorTypes>... predecessors)
{
	auto vector = hlp::makeQVector<
		std::shared_ptr<const DataFilter<ElementTypeOf_t<_PredecessorTypes...>, DimensionalityOf_t<_PredecessorTypes...>::value>>>(
		std::move(predecessors)...);
	return makeSwitch(std::move(vector));
}

} // namespace fs
} // namespace filter

#endif // FILTER_FS_SWITCH_H

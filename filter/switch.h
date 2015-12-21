#ifndef FITLER_SWITCH_H
#define FITLER_SWITCH_H

#include <cassert>
#include <vector>
#include <array>
#include <atomic>

#include <QList>

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/typetraits.h"

#include "helper/threadsafe.h"

#include "helper/helper.h"
#include "helper/qt/vector.h"

namespace filter
{

struct SwitchControl {
	virtual int selection() const = 0;
	virtual void setSelection(int selection) = 0;
};

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _Dimensionality = 0>
class Switch : public PredecessorVectorFilterBase<DataFilter<_ElementType, _Dimensionality>>,
			   public virtual SwitchControl,
			   public virtual DataFilter<_ElementType, _Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	int m_selection;

protected:
	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> selectedPredecessor() const
	{
		int selection = m_selection;
		auto guard = this->m_predecessors.lockConst();
		return guard.data().value(selection);
	}

public:
	Switch(int selection = 0)
		: m_selection(selection)
	{
	}

	// SwitchControl interface
public:
	virtual int selection() const override
	{
		return m_selection;
	}
	virtual void setSelection(int selection) override
	{
		this->setAndInvalidate(m_selection, selection);
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
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		return predecessor->prepare(progress);
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		predecessor->getData(progress, result);
	}
	virtual ndim::sizes<Dimensionality> prepareConst(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		return predecessor->prepareConst(progress);
	}
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = selectedPredecessor();
		hlp::notNull(predecessor);
		predecessor->getConstData(progress, result);
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _ElementType, size_t _Dimensionality>
std::shared_ptr<Switch<_ElementType, _Dimensionality>> makeSwitch()
{
	return std::make_shared<Switch<_ElementType, _Dimensionality>>();
}

template <typename... _PredecessorTypes>
std::shared_ptr<Switch<ElementTypeOf_t<_PredecessorTypes...>, DimensionalityOf_t<_PredecessorTypes...>::value>> makeSwitch(
	std::shared_ptr<_PredecessorTypes>... predecessors)
{
	auto filter = std::make_shared<Switch<ElementTypeOf_t<_PredecessorTypes...>, DimensionalityOf_t<_PredecessorTypes...>::value>>();
	QVector<std::shared_ptr<const DataFilter<ElementTypeOf_t<_PredecessorTypes...>, DimensionalityOf_t<_PredecessorTypes...>::value>>> vector = {
		predecessors...};
	filter->setPredecessors(std::move(vector));
	return filter;
}

} // namespace filter

#endif // FITLER_SWITCH_H

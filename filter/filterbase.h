#ifndef FILTER_FILTERBASE_H
#define FILTER_FILTERBASE_H

#include "filter/filter.h"

#include <vector>
#include <memory>
#include <stdexcept>
#include "variadic.h"
#include "helper/threadsafe.h"

#include <QVector>

/*!
 * \ingroup Filter
 * @{
 */

namespace filter
{
/*! \brief Base implementation for a filter.
 *
 * This base class implements all virtual methods from \ref Predecessor and \ref Successor.
 * When any predecessor notifies an instance of this class about invalidation, all Successors are invalidated.
 *
 * To acutally provide data, inherit the corresponding \ref DataFilter or \ref NoConstDataFilter.
 * */
class FilterBase : public std::enable_shared_from_this<Successor>, public virtual Predecessor, public virtual Successor
{
	mutable std::vector<std::weak_ptr<Successor>> m_successors;

public:
	virtual ~FilterBase() = default;

protected:
	/*! \brief Notifies all successors about invalidation.
	 * */
	void invalidate()
	{
		auto invalidateOrRemove = [this](const std::weak_ptr<Successor> &weak) {
			auto successor = weak.lock();
			if (!successor)
				return true;
			successor->predecessorInvalidated(this);
			return false;
		};

		m_successors.erase(std::remove_if(m_successors.begin(), m_successors.end(), invalidateOrRemove), m_successors.cend());
	}

	template <typename _ParameterType, typename _ValueType>
	void setAndInvalidate(_ParameterType &parameter, _ValueType &&value)
	{
		if (parameter == value)
			return;
		this->invalidate();
		parameter = std::move(value);
	}

	// Invalidatable interface
public:
	virtual void predecessorInvalidated(const Predecessor *) override
	{
		invalidate();
	}

	// Filter interface
public:
	virtual void addSuccessor(std::weak_ptr<Successor> successor) const override
	{
		m_successors.push_back(std::move(successor));
	}

	virtual void removeSuccessor(Successor *successor) const override
	{
		auto removeIfMatchOrNull = [successor](const std::weak_ptr<Successor> &weak) { return weak.lock().get() == successor; };

		m_successors.erase(std::remove_if(m_successors.begin(), m_successors.end(), removeIfMatchOrNull), m_successors.cend());
	}
};

template <typename _ElementType, size_t _Dimensionality = 0>
class NoConstDataFilter : public virtual DataFilter<_ElementType, _Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepareConst(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		return this->prepare(progress);
	}
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		Container<ElementType, Dimensionality> container;
		if (result.ownsData())
			container.reset(result.mutablePointer());
		this->getData(progress, container);
		if (container.pointer().data != result.pointer().data)
			result = std::move(container);
	}
};

template <typename _FilterType>
class PredecessorStore
{
public:
	using FilterType = const _FilterType;

private:
	hlp::Threadsafe<std::shared_ptr<FilterType>> m_predecessor;

public:
	/*!
	 * \brief Gets the stored predecessor.
	 * \return The stored predecessor.
	 */
	std::shared_ptr<FilterType> get() const
	{
		return m_predecessor.get();
	}

	/*!
	 * \brief Replaces the stored predecessor by the given predecessor and (un-)registers the given successor accordingly.
	 * \param predecessor The new predecessor
	 * \param successor The successor to be (un-)registered.
	 *
	 * #### Threading
	 * This method is called only from the \b main thread.
	 */
	std::shared_ptr<FilterType> reset(std::shared_ptr<FilterType> predecessor, const std::shared_ptr<Successor> &successor)
	{
		auto newPredecessor = predecessor.get();
		{
			auto guard = m_predecessor.lock();
			guard.data().swap(predecessor);
		}
		if (predecessor) // oldPredecessor
			predecessor->removeSuccessor(successor.get());
		if (newPredecessor)
			newPredecessor->addSuccessor(successor);
		return predecessor; // Don't use std::move, since it would prevent RVO.
	}
	std::shared_ptr<FilterType> clear(Successor *successor)
	{
		std::shared_ptr<FilterType> predecessor(nullptr);
		{
			auto guard = m_predecessor.lock();
			guard.data().swap(predecessor);
		}
		if (predecessor) // oldPredecessor
			predecessor->removeSuccessor(successor);
		return predecessor; // Don't use std::move, since it would prevent RVO.
	}

	hlp::ThreadsafeGuard<std::shared_ptr<FilterType>> lock()
	{
		return m_predecessor.lock();
	}
	hlp::ThreadsafeGuard<const std::shared_ptr<FilterType>> lockConst() const
	{
		return m_predecessor.lockConst();
	}
};

template <typename _FilterType>
class PredecessorVectorStore
{
public:
	using FilterType = const _FilterType;

private:
	hlp::Threadsafe<QVector<std::shared_ptr<FilterType>>> m_predecessors;

public:
	QVector<std::shared_ptr<FilterType>> get() const
	{
		return m_predecessors.get();
	}

	void reset(QVector<std::shared_ptr<FilterType>> predecessors, const std::shared_ptr<Successor> &successor)
	{
		QVector<std::shared_ptr<FilterType>> newPredecessors = predecessors;
		{
			auto guard = m_predecessors.lock();
			guard.data().swap(predecessors);
		}
		for (const auto &oldPredecessor : predecessors)
			if (oldPredecessor)
				oldPredecessor->removeSuccessor(successor.get());
		for (auto newPredecessor : newPredecessors)
			if (newPredecessor)
				newPredecessor->addSuccessor(successor);
	}
	void clear(Successor *successor)
	{
		QVector<std::shared_ptr<FilterType>> predecessors;
		{
			auto guard = m_predecessors.lock();
			guard.data().swap(predecessors);
		}
		for (const auto &oldPredecessor : predecessors)
			if (oldPredecessor)
				oldPredecessor->removeSuccessor(successor);
	}

	hlp::ThreadsafeGuard<QVector<std::shared_ptr<FilterType>>> lock()
	{
		return m_predecessors.lock();
	}
	hlp::ThreadsafeGuard<const QVector<std::shared_ptr<FilterType>>> lockConst() const
	{
		return m_predecessors.lockConst();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _FilterType>
class SinglePredecessorFilterBase : public FilterBase, public virtual Successor
{
public:
	using FilterType = const _FilterType;

protected:
	PredecessorStore<FilterType> m_predecessor;

public:
	~SinglePredecessorFilterBase()
	{
		m_predecessor.clear(this);
	}

	std::shared_ptr<FilterType> predecessor() const
	{
		return m_predecessor.get();
	}

	void setPredecessor(std::shared_ptr<FilterType> predecessor)
	{
		this->invalidate();
		m_predecessor.reset(predecessor, this->shared_from_this());
	}
};

template <typename _FilterType>
class PredecessorVectorFilterBase : public FilterBase, public virtual Successor
{
public:
	using FilterType = const _FilterType;

protected:
	PredecessorVectorStore<FilterType> m_predecessors;

public:
	~PredecessorVectorFilterBase()
	{
		m_predecessors.clear(this);
	}

	QVector<std::shared_ptr<FilterType>> predecessors() const
	{
		return m_predecessors.get();
	}

	void setPredecessors(QVector<std::shared_ptr<FilterType>> predecessors)
	{
		this->invalidate();
		m_predecessors.reset(predecessors, this->shared_from_this());
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

} // namespace filter

//! @}

#endif // FILTER_FILTERBASE_H

#ifndef FILTER_FILTERBASE_H
#define FILTER_FILTERBASE_H

#include "fc/filter.h"

#include <vector>
#include <memory>
#include <stdexcept>

#include "helper/variadic.h"
#include "helper/threadsafe.h"
#include "variadic.h"

#include <QVector>

/*!
 * \ingroup Filter
 * @{
 */

namespace fc
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
				return true; // Remove weak pointers to successors that don't exist anymore.
			successor->predecessorInvalidated(this);
			return false;
		};

		m_successors.erase(std::remove_if(m_successors.begin(), m_successors.end(), invalidateOrRemove), m_successors.end());
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
		auto item = std::find_if(
			m_successors.begin(), m_successors.end(), [successor](const std::weak_ptr<Successor> &weak) { return weak.lock().get() == successor; });
		if (item != m_successors.cend()) {
			item->swap(m_successors.back());
			m_successors.pop_back();
		}

		auto removeIfNull = [successor](const std::weak_ptr<Successor> &weak) { return weak.expired(); };
		m_successors.erase(std::remove_if(m_successors.begin(), m_successors.end(), removeIfNull), m_successors.end());
	}

protected:
	template <typename PredecessorType>
	void registerAsSuccessor(const std::shared_ptr<PredecessorType> &predecessor)
	{
		if (predecessor)
			predecessor->addSuccessor(this->shared_from_this());
	}
	template <typename PredecessorType>
	void unregisterAsSuccessor(const std::shared_ptr<PredecessorType> &predecessor)
	{
		if (predecessor)
			predecessor->removeSuccessor(this);
	}

	template <typename... PredecessorTypes>
	void registerAsSuccessor(const std::tuple<PredecessorTypes...> &predecessors)
	{
		std::weak_ptr<Successor> self = this->shared_from_this();
		auto op = [&self](const std::shared_ptr<const Predecessor> &predecessor) {
			if (predecessor)
				predecessor->addSuccessor(self);
		};
		hlp::variadic::forEachInTuple(op, predecessors);
	}
	template <typename... PredecessorTypes>
	void unregisterAsSuccessor(const std::tuple<PredecessorTypes...> &predecessors)
	{
		auto op = [this](const std::shared_ptr<const Predecessor> &predecessor) {
			if (predecessor)
				predecessor->removeSuccessor(this);
		};
		hlp::variadic::forEachInTuple(op, predecessors);
	}

	template <typename PredecessorType>
	void registerAsSuccessor(const QVector<std::shared_ptr<PredecessorType>> &predecessors)
	{
		std::weak_ptr<Successor> self = this->shared_from_this();
		for (const std::shared_ptr<PredecessorType> &predecessor : predecessors)
			if (predecessor)
				predecessor->addSuccessor(self);
	}
	template <typename PredecessorType>
	void unregisterAsSuccessor(const QVector<std::shared_ptr<PredecessorType>> &predecessors)
	{
		for (const std::shared_ptr<PredecessorType> &predecessor : predecessors)
			if (predecessor)
				predecessor->removeSuccessor(this);
	}
};

template <typename... PredecessorTypes>
class FilterHandlerBase
{
public:
	using PredecessorTuple = std::tuple<std::shared_ptr<const PredecessorTypes>...>;

	PredecessorTuple predecessors;

	template <size_t I = 0>
	typename std::tuple_element<I, PredecessorTuple>::type &predecessor()
	{
		return std::get<0>(predecessors);
	}
	template <size_t I = 0>
	const typename std::tuple_element<I, PredecessorTuple>::type &predecessor() const
	{
		return std::get<0>(predecessors);
	}
};

template <typename _HandlerType>
class HandlerFilterBase : public FilterBase
{
public:
	using HandlerType = _HandlerType;

	using PredecessorTuple = typename HandlerType::PredecessorTuple;

protected:
	hlp::Threadsafe<HandlerType> m_handler;

public:
	template <typename... Args>
	HandlerFilterBase(Args &&... handlerArgs)
		: m_handler(std::forward<Args>(handlerArgs)...)
	{
	}
	~HandlerFilterBase()
	{
		this->unregisterAsSuccessor(m_handler.unguarded().predecessors);
	}

	PredecessorTuple predecessors() const
	{
		return m_handler.unguarded().predecessors;
	}
	template <size_t I = 0>
	typename std::tuple_element<I, PredecessorTuple>::type predecessor() const
	{
		return m_handler.unguarded().predecessor<I>();
	}

	void setPredecessors(PredecessorTuple predecessors)
	{
		if (m_handler.unguarded().predecessors == predecessors)
			return;
		this->invalidate();
		this->unregisterAsSuccessor(m_handler.unguarded().predecessors);
		this->registerAsSuccessor(predecessors);
		m_handler.lock().data().predecessors = predecessors;
	}
	template <size_t I = 0>
	void setPredecessor(typename std::tuple_element<I, PredecessorTuple>::type predecessor)
	{
		if (std::get<I>(m_handler.unguarded().predecessors) == predecessor)
			return;
		this->invalidate();
		this->unregisterAsSuccessor(std::get<I>(m_handler.unguarded().predecessors));
		this->registerAsSuccessor(predecessor);
		std::get<I>(m_handler.lock().data().predecessors) = predecessor;
	}

	HandlerType handler() const
	{
		return m_handler.get();
	}
	void setHandler(HandlerType handler)
	{
		this->invalidate();
		this->unregisterAsSuccessor(m_handler.unguarded().predecessors);
		this->registerAsSuccessor(handler.predecessors);
		m_handler.lock().data() = handler;
	}
};

} // namespace fc

//! @}

#endif // FILTER_FILTERBASE_H

#ifndef THREADSAFE_H
#define THREADSAFE_H

#include <utility>	// std::move, std::forward
#include <mutex>	  // std::mutex
#include <functional> // std::reference_wrapper

namespace hlp
{

/*!
 * \remarks Can not use std::lock_guard, since ThreadsafeGuard should be moveable and lock_guard is not moveable.
 * */
template <typename _ValueType>
class ThreadsafeGuard
{
public:
	using ValueType = _ValueType;

private:
	std::mutex *m_mutex;
	std::reference_wrapper<ValueType> m_data;

public:
	ThreadsafeGuard(std::mutex &mutex, std::reference_wrapper<ValueType> data)
		: m_mutex(&mutex)
		, m_data(data)
	{
		mutex.lock();
	}
	~ThreadsafeGuard()
	{
		if (m_mutex)
			m_mutex->unlock();
	}

	ThreadsafeGuard(ThreadsafeGuard<ValueType> &&other)
		: m_mutex(other.m_mutex)
		, m_data(other.m_data)
	{
		other.m_mutex = nullptr;
	}
	ThreadsafeGuard &operator=(ThreadsafeGuard<ValueType> &&other)
	{
		m_mutex = other.m_mutex;
		other.m_mutex = nullptr;
		m_data = other.m_data;
	}

	ThreadsafeGuard(const ThreadsafeGuard<ValueType> &) = delete;
	ThreadsafeGuard &operator=(const ThreadsafeGuard<ValueType> &) = delete;

	ValueType &data() const
	{
		return m_data;
	}
};

/*! \brief A threadsafe container for arbitrary data.
 *
 * Acces to the data is sequenced by a std::mutex.
 */
template <typename _ValueType>
class Threadsafe
{
public:
	using ValueType = _ValueType;

private:
	mutable std::mutex m_mutex;
	ValueType m_data;

public:
	template <typename... _CtorTypes>
	Threadsafe(_CtorTypes &&... values)
		: m_data(std::forward<_CtorTypes>(values)...)
	{
	}

	ThreadsafeGuard<ValueType> lock()
	{
		return ThreadsafeGuard<ValueType>(m_mutex, m_data);
	}

	ThreadsafeGuard<const ValueType> lockConst() const
	{
		return ThreadsafeGuard<const ValueType>(m_mutex, m_data);
	}

	ValueType take() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		jfh::unused_variable(guard);

		return std::move(m_data);
	}
	ValueType get() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		jfh::unused_variable(guard);

		return m_data;
	}

	template <typename _AssignType>
	Threadsafe<ValueType> &operator=(_AssignType &&value)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		jfh::unused_variable(lock);

		m_data = std::forward<_AssignType>(value);

		return *this;
	}

	template <typename _CompareType>
	bool operator==(const _CompareType &other) const
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		jfh::unused_variable(lock);

		return m_data == other;
	}
};

} // namespace hlp

#endif // THREADSAFE_H

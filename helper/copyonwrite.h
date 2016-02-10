#ifndef COPYONWRITE
#define COPYONWRITE

#include <atomic>
#include <utility>

#include <memory>

namespace hlp
{

template <typename _ValueType>
class CopyOnWrite
{
public:
	using ValueType = _ValueType;

private:
	std::shared_ptr<ValueType> m_data;

public:
	CopyOnWrite() = default;
	CopyOnWrite(const CopyOnWrite<ValueType> &other) = default;
	CopyOnWrite(CopyOnWrite<ValueType> &&other) = default;

	CopyOnWrite<ValueType> &operator=(const CopyOnWrite<ValueType> &other) = default;
	CopyOnWrite<ValueType> &operator=(CopyOnWrite<ValueType> &&other) = default;

	template <typename... Args>
	void emplace(Args &&... args)
	{
		if (m_data && m_data.unique())
			*m_data = ValueType(std::forward<Args>(args)...);
		else
			m_data = std::make_shared<ValueType>(std::forward<Args>(args)...);
	}

	CopyOnWrite<ValueType> &operator=(const ValueType &value)
	{
		if (m_data && m_data.unique())
			*m_data = value;
		else
			emplace(value);
	}

	CopyOnWrite<ValueType> &operator=(ValueType &&value)
	{
		if (m_data && m_data.unique())
			*m_data = std::move(value);
		else
			emplace(std::move(value));
	}

	ValueType &write()
	{
		if (!m_data)
			m_data = std::make_shared<ValueType>();
		else if (!m_data.unique())
			m_data = std::make_shared<ValueType>(*m_data);
		return *m_data;
	}

	const ValueType &operator*() const
	{
		return *m_data;
	}
	const ValueType *operator->() const
	{
		return m_data.get();
	}
	explicit operator bool() const
	{
		return m_data;
	}

	void reset()
	{
		m_data.reset();
	}
	bool isUnique() const
	{
		return m_data.unique();
	}

	std::shared_ptr<const ValueType> getShared() const
	{
		return m_data;
	}
};

} // namespace hlp

#endif // COPYONWRITE

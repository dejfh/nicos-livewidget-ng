#ifndef FILTER_INPUT_H
#define FILTER_INPUT_H

#include "filter/filter.h"
#include "filter/filterbase.h"

#include "helper/threadsafe.h"

namespace filter
{

template <typename _ValueType>
class InputFilter : public FilterBase, public virtual NoConstDataFilter<DefaultFilterTypeTraits<_ValueType>>
{
public:
	using ValueType = _ValueType;

private:
	hlp::Threadsafe<ValueType> m_data;

public:
	InputFilter()
	{
	}
	template <typename _InitType>
	InputFilter(_InitType &&value)
		: m_data(std::forward<_InitType>(value))
	{
	}

	template <typename _ForwardType>
	void setData(_ForwardType &&data)
	{
		this->invalidate();
		m_data = std::forward<_ForwardType>(data);
	}
	ValueType data() const
	{
		return m_data;
	}

	// DataFilter interface
public:
	virtual void prepare(AsyncProgress &, DurationCounter &, MetaType &) const override
	{
	}
	virtual void getData(ValidationProgress &, CopyType data) const override
	{
		data = m_data;
	}
};

} // namespace filter

#endif // FILTER_INPUT_H

#ifndef SAFECAST_H
#define SAFECAST_H

#include <limits>
#include <type_traits>

#include <cassert>

namespace jfh
{

template <typename... unused_t>
inline void unused_variable(unused_t &...)
{
}

inline void assert_result(bool b)
{
	unused_variable(b);
	assert(b);
}

template <typename in_t, typename out_t, bool is_in_signed = std::numeric_limits<in_t>::is_signed, bool is_out_signed = std::numeric_limits<out_t>::is_signed>
struct __assert_cast_helper {
	static inline void call(in_t value)
	{
		assert(value >= std::numeric_limits<out_t>::lowest() && value <= std::numeric_limits<out_t>::max());
		unused_variable(value);
	}
};

template <typename in_t, typename out_t>
struct __assert_cast_helper<in_t, out_t, true, false> {
	static inline void call(in_t value)
	{
		typedef typename std::make_unsigned<in_t>::type in_ust;
		assert(value >= 0 && in_ust(value) <= std::numeric_limits<out_t>::max());
		unused_variable(value);
	}
};

template <typename in_t, typename out_t>
struct __assert_cast_helper<in_t, out_t, false, true> {
	static inline void call(in_t value)
	{
		typedef typename std::make_unsigned<out_t>::type out_ust;
		assert(value <= out_ust(std::numeric_limits<out_t>::max()));
		unused_variable(value);
	}
};

template <typename out_t, typename in_t>
inline out_t assert_cast(in_t value)
{
	__assert_cast_helper<in_t, out_t>::call(value);
	return out_t(value);
}

template <typename out_t, typename in_t>
inline out_t cast_over_void(in_t *ptr)
{
	return static_cast<out_t>(static_cast<void *>(ptr));
}
template <typename out_t, typename in_t>
inline out_t cast_over_void(const in_t *ptr)
{
	return static_cast<out_t>(static_cast<const void *>(ptr));
}

} // namespace jfh

#endif // SAFECAST_H

#ifndef HELPER_ASSERTCAST_H
#define HELPER_ASSERTCAST_H

#include <limits>
#include <type_traits>

#include <cassert>

#include "helper/helper.h"

namespace hlp
{

template <typename in_t, typename out_t, bool is_in_signed = std::numeric_limits<in_t>::is_signed,
    bool is_out_signed = std::numeric_limits<out_t>::is_signed>
struct __assert_cast_helper {
    static inline void call(in_t value)
    {
        assert(value >= std::numeric_limits<out_t>::lowest() && value <= std::numeric_limits<out_t>::max());
        hlp::unused(value);
    }
};

template <typename in_t, typename out_t>
struct __assert_cast_helper<in_t, out_t, true, false> {
    static inline void call(in_t value)
    {
        assert(value >= 0 && typename std::make_unsigned<in_t>::type(value) <= std::numeric_limits<out_t>::max());
        hlp::unused(value);
    }
};

template <typename in_t, typename out_t>
struct __assert_cast_helper<in_t, out_t, false, true> {
    static inline void call(in_t value)
    {
        assert(value <= typename std::make_unsigned<out_t>::type(std::numeric_limits<out_t>::max()));
        unused_variable(value);
    }
};

template <typename out_t, typename in_t>
inline out_t assert_cast(in_t value)
{
    __assert_cast_helper<in_t, out_t>::call(value);
    return out_t(value);
}

} // namespace hlp

#endif // HELPER_ASSERTCAST_H

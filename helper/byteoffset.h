#ifndef HELPER_BYTEOFFSET_H
#define HELPER_BYTEOFFSET_H

#include <type_traits>

#include "helper/helper.h"

#include "boost/type_traits/copy_cv.hpp"

namespace hlp
{

struct byte_offset_t {
    ptrdiff_t value;

    byte_offset_t()
    {
    }
    explicit byte_offset_t(ptrdiff_t value)
        : value(value)
    {
    }
    byte_offset_t(const byte_offset_t &other) = default;
    byte_offset_t &operator=(const byte_offset_t &other)
    {
        value = other.value;
        return *this;
    }

    operator bool() const
    {
        return value;
    }

    template <typename Type>
    static byte_offset_t inArray()
    {
        typename std::remove_cv<Type>::type *t0 = nullptr;
        typename std::remove_cv<Type>::type *t1 = &t0[1];
        char *c0 = hlp::cast_over_void<char *>(t0);
        char *c1 = hlp::cast_over_void<char *>(t1);
        return byte_offset_t(c1 - c0);
    }
};

inline static byte_offset_t operator-(byte_offset_t offset)
{
    return byte_offset_t(-offset.value);
}
inline static bool operator!(byte_offset_t offset)
{
    return !offset.value;
}

inline static byte_offset_t operator+(byte_offset_t a, byte_offset_t b)
{
    return byte_offset_t(a.value + b.value);
}
inline static byte_offset_t &operator+=(byte_offset_t &a, byte_offset_t b)
{
    a.value += b.value;
    return a;
}

inline static byte_offset_t operator-(byte_offset_t a, byte_offset_t b)
{
    return byte_offset_t(a.value - b.value);
}
inline static byte_offset_t &operator-=(byte_offset_t &a, byte_offset_t b)
{
    a.value -= b.value;
    return a;
}

/*
 * ===== Pointer Arithmetic =====
 * */

template <typename _T>
static _T *operator+(_T *ptr, byte_offset_t offset)
{
    using char_cv_t = typename boost::copy_cv<char, _T>::type;
    auto *char_ptr = hlp::cast_over_void<char_cv_t *>(ptr);
    char_ptr += offset.value;
    return hlp::cast_over_void<_T *>(char_ptr);
}
template <typename _T>
static _T *operator+(byte_offset_t offset, _T *ptr)
{
    return ptr + offset;
}
template <typename _T>
static _T *operator-(_T *ptr, byte_offset_t offset)
{
    return ptr + (-offset);
}

template <typename _T>
static _T *&operator+=(_T *&ptr, byte_offset_t offset)
{
    ptr = ptr + offset;
    return ptr;
}
template <typename _T>
static _T *&operator-=(_T *&ptr, byte_offset_t offset)
{
    return ptr += (-offset);
}

/*
 * ===== Number Arithmetic =====
 * */

// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
// operator+(byte_offset_t offset, ValueType value)
//{
//    return byte_offset_t(offset.value + value);
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
// operator+(ValueType value, byte_offset_t offset)
//{
//    return byte_offset_t(value + offset.value);
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t &>::type //
// operator+=(byte_offset_t &offset, ValueType value)
//{
//    offset.value += value;
//    return offset;
//}

// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
// operator-(byte_offset_t offset, ValueType value)
//{
//    return byte_offset_t(offset.value - value);
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
// operator-(ValueType value, byte_offset_t offset)
//{
//    return byte_offset_t(value - offset.value);
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t &>::type //
// operator-=(byte_offset_t &offset, ValueType value)
//{
//    offset.value -= value;
//    return offset;
//}

template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
operator*(byte_offset_t offset, ValueType value)
{
    return byte_offset_t(offset.value * value);
}
template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
operator*(ValueType value, byte_offset_t offset)
{
    return byte_offset_t(value * offset.value);
}
template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t &>::type //
operator*=(byte_offset_t &offset, ValueType value)
{
    offset.value *= value;
    return offset;
}

template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
operator/(byte_offset_t offset, ValueType value)
{
    return byte_offset_t(offset.value / value);
}
template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t>::type //
operator/(ValueType value, byte_offset_t offset)
{
    return byte_offset_t(value / offset.value);
}
template <typename ValueType>
static typename std::enable_if<std::is_arithmetic<ValueType>::value, byte_offset_t &>::type //
operator/=(byte_offset_t &offset, ValueType value)
{
    offset.value /= value;
    return offset;
}

// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, bool>::type //
// operator==(byte_offset_t offset, ValueType value)
//{
//    return offset.value == value;
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, bool>::type //
// operator==(ValueType value, byte_offset_t offset)
//{
//    return value == offset.value;
//}

// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, bool>::type //
// operator!=(byte_offset_t offset, ValueType value)
//{
//    return offset.value != value;
//}
// template <typename ValueType>
// static typename std::enable_if<std::is_arithmetic<ValueType>::value, bool>::type //
// operator!=(ValueType value, byte_offset_t offset)
//{
//    return value != offset.value;
//}

} // namespace hlp

#endif // HELPER_BYTEOFFSET_H

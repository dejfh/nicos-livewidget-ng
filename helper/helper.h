#ifndef HELPER_HELPER_H
#define HELPER_HELPER_H

#include <memory>

#include <cassert>

#include "helper/type_traits.h"

namespace hlp
{

/**
 * @brief Suppress compiler warnings for given unreferenced variables.
 *
 * Especially usefull for lock guards like std::lock_guard, hlp::ThreadsafeGuard or hlp::python::Gil.
 */
template <typename... UnusedTypes>
void unused(UnusedTypes &&...)
{
}

inline void assert_result(bool result)
{
    if (!result) {
        assert(result);
    }
	unused(result);
}

struct assert_true {
    assert_true() = default;
    assert_true(bool value)
    {
        assert_result(value);
    }
};

inline const assert_true &operator<<(const assert_true &at, bool value)
{
	assert_result(value);
    return at;
}
inline assert_true &&operator<<(assert_true &&at, bool value)
{
	assert_result(value);
    return std::move(at);
}
inline const assert_true &operator>>(bool value, const assert_true &at)
{
	assert_result(value);
    return at;
}
inline assert_true &&operator>>(bool value, assert_true &&at)
{
	assert_result(value);
    return std::move(at);
}

template <typename out_t, typename in_t>
inline out_t cast_over_void(in_t *ptr)
{
	using void_cv_t = hlp::copy_cv_t<void, in_t>;
    return static_cast<out_t>(static_cast<void_cv_t *>(ptr));
}

/*! \brief Throws a std::invalid_argument exception if the given pointer is null.
 *
 * \param pointer The pointer to check for null.
 * \returns The given pointer if it is not null.
 * */
template <typename Type>
Type *throwIfNull(Type *pointer)
{
	if (!pointer)
		throw std::invalid_argument("Pointer is null.");
	return pointer;
}

/*! \brief Throws a std::invalid_argument exception if the given pointer is null.
 *
 * \param pointer The pointer to check for null.
 * \returns The given pointer if it is not null.
 * */
template <typename Type>
Type *throwIfNull(const std::unique_ptr<Type> &pointer)
{
	if (!pointer)
		throw std::invalid_argument("Pointer is null.");
    return pointer.get();
}

template <typename Type>
void throwIfNull(std::unique_ptr<Type> &&) = delete;

/*! \brief Throws a std::invalid_argument exception if the given pointer is null.
 *
 * \param pointer The pointer to check for null.
 * \returns The given pointer if it is not null.
 * */
template <typename Type>
Type *throwIfNull(const std::shared_ptr<Type> &pointer)
{
	if (!pointer)
		throw std::invalid_argument("Pointer is null.");
    return pointer.get();
}

template <typename Type>
void throwIfNull(std::shared_ptr<Type> &&) = delete;

template <typename Type>
const Type &constRef(Type &&) = delete;

template <typename Type>
const Type &constRef(Type &ref)
{
	return ref;
}

template <typename Out, typename In>
typename std::enable_if<!std::is_assignable<Out &, In &&>::value, Out>::type castIfAssignable(In &&)
{
	return Out();
}
template <typename Out, typename In>
typename std::enable_if<std::is_assignable<Out &, In &&>::value, Out>::type castIfAssignable(In &&in)
{
	return std::forward<In>(in);
}

template <typename Out, typename In>
typename std::enable_if<!std::is_assignable<Out &, In &&>::value>::type assignIfAssignable(Out &, In &&)
{
	// Not assignable;
}
template <typename Out, typename In>
typename std::enable_if<std::is_assignable<Out &, In &&>::value>::type assignIfAssignable(Out &out, In &&in)
{
	out = std::forward<In>(in);
}

} // namespace hlp

#endif // HELPER_HELPER_H

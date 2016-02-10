#ifndef HELPER_HELPER_H
#define HELPER_HELPER_H

#include <memory>

#include <cassert>

namespace hlp
{

/// @brief Suppress compiler warnings for given unused variables.
template <typename... UnusedTypes>
void unused(UnusedTypes &&...)
{
}

inline void assert_result(bool result)
{
	assert(result);
	unused(result);
}

struct assert_true {
};

inline void operator<<(const assert_true &, bool value)
{
	assert_result(value);
}
inline void operator<<(assert_true &&, bool value)
{
	assert_result(value);
}
inline void operator>>(bool value, const assert_true &)
{
	assert_result(value);
}
inline void operator>>(bool value, assert_true &&)
{
	assert_result(value);
}

/*! \brief Throws a std::invalid_argument exception if the given pointer is null.
 *
 * \param pointer The pointer to check for null.
 * \returns The given pointer if it is not null.
 * */
template <typename _Type>
_Type *notNull(_Type *pointer)
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
template <typename _Type>
_Type *notNull(const std::unique_ptr<_Type> &pointer)
{
	if (!pointer)
		throw std::invalid_argument("Pointer is null.");
	return pointer.get();
}

template <typename _Type>
_Type *notNull(std::unique_ptr<_Type> &&) = delete;

/*! \brief Throws a std::invalid_argument exception if the given pointer is null.
 *
 * \param pointer The pointer to check for null.
 * \returns The given pointer if it is not null.
 * */
template <typename _Type>
_Type *notNull(const std::shared_ptr<_Type> &pointer)
{
	if (!pointer)
		throw std::invalid_argument("Pointer is null.");
	return pointer.get();
}

template <typename _Type>
_Type *notNull(std::shared_ptr<_Type> &&) = delete;

template <typename _Type>
const _Type &asConst(_Type &&) = delete;
template <typename _Type>
const _Type &asConst(_Type &ref)
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

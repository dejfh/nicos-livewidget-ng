#ifndef HELPER_HELPER_H
#define HELPER_HELPER_H

#include <memory>

namespace hlp
{

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

template <typename... UnusedTypes>
void unused(UnusedTypes &&...)
{
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

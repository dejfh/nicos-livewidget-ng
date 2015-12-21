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

} // namespace hlp

#endif // HELPER_HELPER_H

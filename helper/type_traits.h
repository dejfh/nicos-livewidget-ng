#ifndef HELPER_TYPETRAITS_H
#define HELPER_TYPETRAITS_H

namespace hlp
{

template <typename A, typename B>
struct copy_cv {
	using type = A;
};

template <typename A, typename B>
struct copy_cv<A, const B> {
	using type = const A;
};

template <typename A, typename B>
struct copy_cv<A, volatile B> {
	using type = volatile A;
};

template <typename A, typename B>
struct copy_cv<A, const volatile B> {
	using type = const volatile A;
};

template <typename A, typename B>
using copy_cv_t = typename copy_cv<A, B>::type;

} // namespace hlp

#endif // HELPER_TYPETRAITS_H

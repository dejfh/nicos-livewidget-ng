#ifndef JFH_VARIADIC_H
#define JFH_VARIADIC_H

#include <cstddef>
#include <tuple>
#include <array>
#include <type_traits>

namespace jfh
{

namespace variadic
{

// nth_type
template <size_t N, typename _Type, typename... _Types>
struct nth_type {
	typedef typename nth_type<N - 1, _Types...>::type type;
};

template <typename _Type, typename... _Types>
struct nth_type<0, _Type, _Types...> {
	typedef _Type type;
};

template <size_t N, typename... _Types>
using nth_type_t = typename nth_type<N, _Types...>::type;

// sequence
template <size_t... S>
struct Sequence {
};

template <size_t N, size_t... S>
struct SequenceBuilder {
	typedef typename SequenceBuilder<N - 1, N - 1, S...>::type type;
};

template <size_t... S>
struct SequenceBuilder<0, S...> {
	typedef Sequence<S...> type;
};

template <size_t N>
typename SequenceBuilder<N>::type makeSequence()
{
	return typename SequenceBuilder<N>::type();
}

template <typename... Types>
typename SequenceBuilder<sizeof...(Types)>::type makeSequence(const std::tuple<Types...> &)
{
	return typename SequenceBuilder<sizeof...(Types)>::type();
}
template <typename Type, size_t N>
typename SequenceBuilder<N>::type makeSequence(const std::array<Type, N> &)
{
	return typename SequenceBuilder<N>::type();
}

// result_of
template <typename _Operation, typename... _Args>
struct result_of {
	typedef decltype(std::declval<_Operation>()(std::declval<_Args>()...)) type;
};

template <typename _Operation, typename... _Args>
using result_of_t = typename result_of<_Operation, _Args...>::type;

template <typename _Operation, typename... _Args>
using returns_void = std::integral_constant<bool, std::is_void<typename result_of<_Operation, _Args...>::type>::value>;

template <typename _Type>
struct not_void {
	typedef _Type type;
};
template <>
struct not_void<void> {
};

template <typename _Type>
using not_void_t = typename not_void<_Type>::type;

template <typename _Type>
struct only_void {
};
template <>
struct only_void<void> {
	typedef void type;
};

template <typename _Type>
using only_void_t = typename only_void<_Type>::type;

template <typename... _Types>
struct one_type {
};
template <typename _Type>
struct one_type<_Type> {
	typedef _Type type;
};

template <typename... _Types>
using one_type_t = typename one_type<_Types...>::type;

template <typename... _Types>
struct tuple_or_one {
	typedef std::tuple<_Types...> type;

	static std::tuple<_Types...> &fromTuple(std::tuple<_Types...> &tuple)
	{
		return tuple;
	}
	static const std::tuple<_Types...> &fromTuple(const std::tuple<_Types...> &tuple)
	{
		return tuple;
	}

	static std::tuple<_Types...> &toTuple(std::tuple<_Types...> &value)
	{
		return value;
	}
	static const std::tuple<_Types...> &toTuple(const std::tuple<_Types...> &value)
	{
		return value;
	}
};
template <typename _Type>
struct tuple_or_one<_Type> {
	typedef _Type type;

	static _Type &fromTuple(std::tuple<_Type> &tuple)
	{
		return std::get<0>(tuple);
	}
	static const _Type &fromTuple(const std::tuple<_Type> &tuple)
	{
		return std::get<0>(tuple);
	}

	static std::tuple<_Type> toTuple(_Type value)
	{
		return std::tuple<_Type>(value);
	}
};

template <typename... _Types>
using tuple_or_one_t = typename tuple_or_one<_Types...>::type;

// hasSignature
template <typename... _Args>
std::false_type hasSignature(_Args...);

template <typename _Operation, typename... _Args>
auto hasSignature(_Operation, _Args...) -> decltype(std::declval<_Operation>()(std::declval<_Args>()...), std::true_type());

template <typename _Operation, typename... _Args>
struct HasSignatureHelper {
	typedef decltype(hasSignature(std::declval<_Operation>(), std::declval<_Args>()...)) type;
};

template <typename _Operation, typename... _Args>
struct HasSignature : HasSignatureHelper<_Operation, _Args...>::type {
};

// any
inline bool any()
{
	return false;
}
template <typename... _Args>
bool any(bool arg, _Args &&... moreArgs)
{
	return arg || any(moreArgs...);
}

// all
inline bool all()
{
	return true;
}
template <typename... _Args>
bool all(bool arg, _Args... moreArgs)
{
	return arg && all(moreArgs...);
}

// accumulate
template <typename _Result, typename _Operation>
_Result accumulate(_Result init, _Operation)
{
	return init;
}
template <typename _Result, typename _Operation, typename _Arg, typename... _MoreArgs>
_Result accumulate(_Result init, _Operation op, _Arg arg, _MoreArgs... moreArgs)
{
	init = op(init, std::get<0>(arg));
	return accumulate(init, op, moreArgs...);
}
template <typename _Result, typename _Operation, typename _Args, size_t... I>
_Result _accumulate(_Result init, _Operation op, _Args args, Sequence<I...>)
{
	return accumulate(init, op, std::get<I>(args)...);
}
template <typename _Result, typename _Operation, typename... _Args>
_Result accumulate(_Result init, _Operation op, const ::std::tuple<_Args...> &args)
{
	return _accumulate(init, op, args, makeSequence(args));
}

// call
template <typename _Result, typename _Operation, typename _Args, size_t... I>
_Result _callR(_Operation op, _Args &args, Sequence<I...>)
{
	return op(std::get<I>(args)...);
}

template <typename _Operation, typename _Args, size_t... I>
void _callV(_Operation op, _Args &args, Sequence<I...>)
{
	op(std::get<I>(args)...);
}

template <typename _Operation, typename... _Args>
auto call(_Operation op, std::tuple<_Args...> &args) //
	-> not_void_t<result_of_t<_Operation, _Args...>>
{
	return _callR<typename result_of<_Operation, _Args...>::type>(op, args, makeSequence(args));
}

template <typename _Operation, typename... _Args>
auto call(_Operation op, std::tuple<_Args...> &args) //
	-> only_void_t<result_of_t<_Operation, _Args...>>
{
	_callV(op, args, makeSequence(args));
}

template <typename _Operation, typename _Arg>
void callEach(_Operation op, _Arg &&arg)
{
	op(std::forward<_Arg>(arg));
}
template <typename _Operation, typename _Arg, typename... _More_Args>
void callEach(_Operation op, _Arg &&arg, _More_Args &&... more_args)
{
	op(std::forward<_Arg>(arg));
	callEach(op, std::forward<_More_Args>(more_args)...);
}

template <typename _Operation, typename _Tuple, size_t... I>
void callEach(_Operation op, _Tuple &tuple, Sequence<I...>)
{
	callEach(op, std::get<I>(tuple)...);
}

} // namespace variadic

} // namespace jfh

#endif // JFH_VARIADIC_H

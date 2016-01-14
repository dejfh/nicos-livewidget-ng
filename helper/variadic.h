#ifndef HELPER_VARIADIC_H
#define HELPER_VARIADIC_H

#include <cstddef>
#include <type_traits>
#include <tuple>
#include <array>

namespace hlp
{

namespace variadic
{

template <typename _Type, size_t _N, _Type _Value0, _Type... _ValueM>
struct NthValue {
	using type = typename NthValue<_Type, _N - 1, _ValueM...>::type;
};
template <typename _Type, _Type _Value0, _Type... _ValueM>
struct NthValue<_Type, 0, _Value0, _ValueM...> {
	using type = std::integral_constant<_Type, _Value0>;
};

template <size_t _N, typename _Type0, typename... _TypeM>
struct NthType {
	using type = typename NthType<_N - 1, _TypeM...>::type;
};
template <typename _Type0, typename... _TypeM>
struct NthType<0, _Type0, _TypeM...> {
	using type = _Type0;
};

template <size_t _N, typename... _Types>
using NthType_t = typename NthType<_N, _Types...>::type;

template <typename _Type, _Type... t_Values>
struct ConstArray;

template <typename _LeftConstArray, typename _RightConstArray>
struct _CatHelper;
template <typename _LeftConstArray, typename _RightType, _RightType... _RightValues>
struct _CatHelper<_LeftConstArray, ConstArray<_RightType, _RightValues...>> {
	using type = typename _LeftConstArray::template append<_RightValues...>;
};

template <typename _ConstArray, typename _IndexArray>
struct _SelectHelper;
template <typename _ConstArray, typename _IndexType, _IndexType... _Indices>
struct _SelectHelper<_ConstArray, ConstArray<_IndexType, _Indices...>> {
	using type = ConstArray<typename _ConstArray::value_type, _ConstArray::template at<_Indices>::value...>;
};

template <size_t _Count, typename _Type, _Type _Start, _Type _Inc, _Type... _PrependValues>
struct _SequenceBuilder;
template <size_t _Count = 0, typename _Type = size_t, _Type _Start = 0, _Type _Inc = 1>
using Sequence = typename _SequenceBuilder<_Count, _Type, _Start, _Inc>::type;

template <size_t... Values>
using ConstIndexArray = ConstArray<size_t, Values...>;

template <typename t_Type, t_Type... t_Values>
struct ConstArray {
	using value_type = t_Type;
	using size = std::integral_constant<size_t, sizeof...(t_Values)>;

	template <size_t I>
	using at = typename NthValue<t_Type, I, t_Values...>::type;

	template <t_Type... _AppendValues>
	using append = ConstArray<t_Type, t_Values..., _AppendValues...>;

	template <t_Type... _PrependValues>
	using prepend = ConstArray<t_Type, _PrependValues..., t_Values...>;

	template <typename _CastType>
	using cast = ConstArray<_CastType, t_Values...>;

	template <typename _CatArray>
	using cat = typename _CatHelper<ConstArray<t_Type, t_Values...>, _CatArray>::type;

	template <typename _IndexArray>
	using select = typename _SelectHelper<ConstArray<t_Type, t_Values...>, _IndexArray>::type;

	using reverse = select<Sequence<size::value, int, size::value - 1, -1>>;
};

template <size_t _Count, typename Type, Type _Start, Type _Inc, Type... _PrependValues>
struct _SequenceBuilder {
	using type = typename _SequenceBuilder<_Count - 1, Type, _Start + _Inc, _Inc, _PrependValues..., _Start>::type;
};
template <typename Type, Type _Start, Type _Inc, Type... _PrependValues>
struct _SequenceBuilder<0, Type, _Start, _Inc, _PrependValues...> {
	using type = ConstArray<Type, _PrependValues...>;
};

template <size_t N>
Sequence<N> makeSequence()
{
	return Sequence<N>();
}
template <typename Type>
Sequence<std::tuple_size<Type>::value> makeSequence(Type &)
{
	return Sequence<std::tuple_size<Type>::value>();
}

/**
 * @brief Forward a value or rvalue as an rvalue and an lvalue as an lvalue
 */
template <size_t I, typename Tuple>
typename std::enable_if<!std::is_lvalue_reference<Tuple>::value,						 //
	typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type && //
	>::type
forwardFromTuple(typename std::remove_reference<Tuple>::type &tuple)
{
	return static_cast<typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type &&>(std::get<I>(tuple));
}

template <size_t I, typename Tuple>
typename std::enable_if<std::is_lvalue_reference<Tuple>::value,							//
	typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type & //
	>::type
forwardFromTuple(Tuple &&tuple)
{
	return static_cast<typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type &>(std::get<I>(tuple));
}

template <size_t I, typename Tuple>
typename std::enable_if<!std::is_lvalue_reference<Tuple>::value,						 //
	typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type && //
	>::type
forwardFromTuple(Tuple &&tuple)
{
	return static_cast<typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type &&>(std::get<I>(tuple));
}

///**
// * @brief Forward an lvalue
// */
// template <size_t I, typename Tuple>
// typename std::tuple_element<I, typename std::remove_reference<Tuple>::type>::type & //
//	forwardFromTuple(typename std::remove_reference<Tuple>::type &&tuple)
//{
//	static_assert(!std::is_lvalue_reference<Tuple>::value, "template argument substituting Tuple is an lvalue reference type");
//	return std::get<I>(tuple);
//}

template <size_t Count>
struct _forEachInTupleCaller {
	template <size_t I, typename Operation, typename... Tuples>
	static void call(Operation op, Tuples &&... args)
	{
		static_assert(I < 1000000, "I is way too large...");
		if (I > 0)
			call<(I ? I - 1 : 0)>(op, std::forward<Tuples>(args)...);
		op(std::get<I>(args)...);
	}
};
template <>
struct _forEachInTupleCaller<0> {
	template <size_t I, typename Operation, typename... Tuples>
	static void call(Operation, Tuples &&...)
	{
	}
};

template <typename Operation, typename... Tuples>
void forEachInTuple(Operation op, Tuples &&... args)
{
	using Tuple0 = NthType_t<0, Tuples...>;
	static const size_t Count = std::tuple_size<typename std::remove_reference<Tuple0>::type>::value;

	_forEachInTupleCaller<Count>::template call<(Count - 1)>(op, std::forward<Tuples>(args)...);
}

// result_of
template <typename Operation, typename... Args>
struct result_of {
	typedef decltype(std::declval<Operation>()(std::declval<Args>()...)) type;
};

template <typename Operation, typename... Args>
using result_of_t = typename result_of<Operation, Args...>::type;

template <typename Operation, typename Tuple, size_t... I>
void _call(Operation &&op, Tuple &&args, ConstIndexArray<I...>)
{
	op(forwardFromTuple<I, Tuple>(args)...);
}

template <typename Operation, typename Tuple, size_t... I>
auto _callR(Operation &&op, Tuple &&args, ConstIndexArray<I...>) //
	-> decltype(op(forwardFromTuple<I, Tuple>(args)...))
{
	return op(forwardFromTuple<I, Tuple>(args)...);
}

template <typename Operation, typename Tuple>
void call(Operation &&op, Tuple &&args)
{
	_call(op, std::forward<Tuple>(args), makeSequence(args));
}

template <typename Operation, typename Tuple>
auto callR(Operation &&op, Tuple &&args) //
	-> decltype(_callR(op, std::forward<Tuple>(args), makeSequence(args)))
{
	return _callR(op, std::forward<Tuple>(args), makeSequence(args));
}

} // namespace variadic

} // namespace hlp

#endif // HELPER_VARIADIC_H

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

template <size_t _N, typename _Type0, typename... _TypeM>
using NthType_t = NthType<_N, _Type0, _TypeM...>;

template <typename _Type, _Type... _Values>
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

template <typename _Type, _Type... _Values>
struct ConstArray {
	using value_type = _Type;
	using size = std::integral_constant<size_t, sizeof...(_Values)>;

	template <size_t I>
	using at = typename NthValue<_Type, I, _Values...>::type;

	template <_Type... _AppendValues>
	using append = ConstArray<_Type, _Values..., _AppendValues...>;

	template <_Type... _PrependValues>
	using prepend = ConstArray<_Type, _PrependValues..., _Values...>;

	template <typename _CastType>
	using cast = ConstArray<_CastType, _Values...>;

	template <typename _CatArray>
	using cat = typename _CatHelper<ConstArray<_Type, _Values...>, _CatArray>::type;

	template <typename _IndexArray>
	using select = typename _SelectHelper<ConstArray<_Type, _Values...>, _IndexArray>::type;

	using reverse = select<Sequence<size::value, int, size::value - 1, -1>>;
};

template <size_t _Count, typename _Type, _Type _Start, _Type _Inc, _Type... _PrependValues>
struct _SequenceBuilder {
	using type = typename _SequenceBuilder<_Count - 1, _Type, _Start + _Inc, _Inc, _PrependValues..., _Start>::type;
};
template <typename _Type, _Type _Start, _Type _Inc, _Type... _PrependValues>
struct _SequenceBuilder<0, _Type, _Start, _Inc, _PrependValues...> {
	using type = ConstArray<_Type, _PrependValues...>;
};

template <size_t N>
Sequence<N> makeSequence()
{
	return Sequence<N>();
}
template <typename _Type>
Sequence<std::tuple_size<_Type>::value> makeSequence(_Type &)
{
	return Sequence<std::tuple_size<_Type>::value>();
}

template <size_t I, typename _Operation, typename... _Tuples>
void _forEachInTuple(_Operation op, _Tuples &&... args)
{
	if (I > 0)
		_forEachInTuple<(I ? I - 1 : 0)>(op, std::forward<_Tuples>(args)...);
	op(std::get<I>(args)...);
}

template <typename _T>
struct blubb;

template <typename _Operation, typename _Tuple0, typename... _TupleN>
void forEachInTuple(_Operation op, _Tuple0 &&arg0, _TupleN &&... argN)
{
	using NoRefTuple = typename std::remove_reference<_Tuple0>::type;
	using TupleSize = std::tuple_size<NoRefTuple>;
	_forEachInTuple<TupleSize::value - 1>(op, std::forward<_Tuple0>(arg0), std::forward<_TupleN>(argN)...);
}

// result_of
template <typename _Operation, typename... _Args>
struct result_of {
	typedef decltype(std::declval<_Operation>()(std::declval<_Args>()...)) type;
};

template <typename _Operation, typename... _Args>
using result_of_t = typename result_of<_Operation, _Args...>::type;

} // namespace variadic

} // namespace hlp

#endif // HELPER_VARIADIC_H

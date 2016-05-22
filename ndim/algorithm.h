#ifndef NDIM_ALGORITHM_H
#define NDIM_ALGORITHM_H

#include <ndim/pointer.h>
#include <stddef.h>

#include <tuple>

#include <ndim/iterator.h>

#include "variadic.h"
#include "ndim/_helper.h"

namespace ndim
{

template <typename _OperationType, size_t _Dimensions, typename... _Types>
auto operate(_OperationType op, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
    -> typename std::enable_if<!jfh::variadic::HasSignature<_OperationType, std::array<size_t, _Dimensions>, _Types...>::value>::type
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = *iterator;
        jfh::variadic::_callV(op, tuple, jfh::variadic::makeSequence(tuple));
	}
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
auto operate(_OperationType op, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
    -> typename std::enable_if<jfh::variadic::HasSignature<_OperationType, std::array<size_t, _Dimensions>, _Types...>::value>::type
// operation with argument for coordinates
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(iterator.coords()), *iterator);
        jfh::variadic::_callV(op, tuple, jfh::variadic::makeSequence(tuple));
	}
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
auto accumulate(_OperationType op, _Result value, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
	->
    typename std::enable_if<!jfh::variadic::HasSignature<_OperationType, _Result, std::array<size_t, _Dimensions>, _Types...>::value, _Result>::type
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(value), *iterator);
        value = jfh::variadic::_callR(op, tuple, jfh::variadic::makeSequence(tuple));
	}
	return value;
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
auto accumulate(_OperationType op, _Result value, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
    -> typename std::enable_if<jfh::variadic::HasSignature<_OperationType, _Result, std::array<size_t, _Dimensions>, _Types...>::value, _Result>::type
// operation with argument for coordinates
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(value, iterator.coords()), *iterator);
        value = jfh::variadic::_callR(op, tuple, jfh::variadic::makeSequence(tuple));
	}
	return value;
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
void operate(_OperationType op, pointer<_Types, _Dimensions>... args)
{
	operate(op, std::make_tuple(args...));
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
_Result accumulate(_OperationType op, _Result value, pointer<_Types, _Dimensions>... args)
{
	return accumulate(op, value, std::make_tuple(args...));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform(pointer<_OutType, _Dimensions> out, _OperationType op, std::tuple<pointer<_InTypes, _Dimensions>...> in)
{
	operate(_helper::makeAssignOperation(out, op, in), std::tuple_cat(std::make_tuple(out), in));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform(pointer<_OutType, _Dimensions> out, _OperationType op, pointer<_InTypes, _Dimensions>... in)
{
	transform(out, op, std::make_tuple(in...));
}

template <typename _OutType, size_t _Dimensions, typename _InType>
void copy(pointer<_InType, _Dimensions> in, pointer<_OutType, _Dimensions> out)
{
	operate([](_InType &in, _OutType &out) { out = in; }, in, out);
}

template <typename _PtrType, size_t _Dimensionality, typename _FillType>
void fill(pointer<_PtrType, _Dimensionality> ptr, const _FillType &value)
{
	operate([&value](_PtrType &out) { out = value; }, ptr);
}

} // namespace ndim

#endif // NDIM_ALGORITHM_H

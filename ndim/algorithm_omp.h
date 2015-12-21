#ifndef NDIM_ALGORITHM_OMP_H
#define NDIM_ALGORITHM_OMP_H

#ifdef _OPENMP

#include <ndim/pointer.h>
#include <stddef.h>

#include <tuple>

#include <ndim/iterator.h>

#include "variadic.h"
#include "ndim/_helper.h"

#include <omp.h>

#ifdef _DEBUG
#include <QDebug>
#endif // _DEBUG

namespace ndim
{

template <size_t _D, typename... _Ts>
size_t jumpIterator_omp(ndim::multiIterator<_D, _Ts...> &iterator, size_t totalCount)
{
#ifdef _DEBUG
	if (!omp_in_parallel())
		qWarning() << "algorithm_omp function called outside of parallel region.";
#endif // _DEBUG

	int threadIndex = omp_get_thread_num();
	int threadCount = omp_get_num_threads();
	size_t start = totalCount * threadIndex / threadCount;
	size_t end = totalCount * (threadIndex + 1) / threadCount;
	size_t count = end - start;
	iterator += start;
	return count;
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
auto operate_omp(_OperationType op, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
	-> typename std::enable_if<!jfh::variadic::HasSignature<_OperationType, std::array<size_t, _Dimensions>, _Types...>::value>::type
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	count = jumpIterator_omp(iterator, count);

	for (size_t i = 0; i < count; ++i, ++iterator){
		auto tuple = *iterator;
		jfh::variadic::_callV(op, tuple, jfh::variadic::makeSequence(tuple));
	}
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
auto operate_omp(_OperationType op, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
	-> typename std::enable_if<jfh::variadic::HasSignature<_OperationType, std::array<size_t, _Dimensions>, _Types...>::value>::type
// operation with argument for coordinates
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	count = jumpIterator_omp(iterator, count);

	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(iterator.coords()), *iterator);
		jfh::variadic::_callV(op, tuple, jfh::variadic::makeSequence(tuple));
	}
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
auto accumulate_omp(_OperationType op, _Result value, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
	-> typename std::enable_if<!jfh::variadic::HasSignature<_OperationType, _Result, std::array<size_t, _Dimensions>, _Types...>::value, _Result>::type
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	count = jumpIterator_omp(iterator, count);

	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(value), *iterator);
		value = jfh::variadic::_callR<_Result>(op, tuple, jfh::variadic::makeSequence(tuple));
	}
	return value;
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
auto accumulate_omp(_OperationType op, _Result value, std::tuple<pointer<_Types, _Dimensions>...> args) // (manual break)
	-> typename std::enable_if<jfh::variadic::HasSignature<_OperationType, _Result, std::array<size_t, _Dimensions>, _Types...>::value, _Result>::type
// operation with argument for coordinates
{
	size_t count = std::get<0>(args).size();
	auto iterator = ndim::makeMultiIterator(args);
	count = jumpIterator_omp(iterator, count);

	for (size_t i = 0; i < count; ++i, ++iterator) {
		auto tuple = std::tuple_cat(std::make_tuple(value, iterator.coords()), *iterator);
		value = jfh::variadic::_callR<_Result>(op, tuple, jfh::variadic::makeSequence(tuple));
	}
	return value;
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
void operate_omp(_OperationType op, pointer<_Types, _Dimensions>... args)
{
	operate_omp(op, std::make_tuple(args...));
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
_Result accumulate_omp(_OperationType op, _Result value, pointer<_Types, _Dimensions>... args)
{
	return accumulate_omp(op, value, std::make_tuple(args...));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform_omp(pointer<_OutType, _Dimensions> out, _OperationType op, std::tuple<pointer<_InTypes, _Dimensions>...> in)
{
	operate_omp(_helper::makeAssignOperation(out, op, in), std::tuple_cat(std::make_tuple(out), in));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform_omp(pointer<_OutType, _Dimensions> out, _OperationType op, pointer<_InTypes, _Dimensions>... in)
{
	transform_omp(out, op, std::make_tuple(in...));
}

template <typename _OutType, size_t _Dimensions, typename _InType>
void copy_omp(pointer<_InType, _Dimensions> in, pointer<_OutType, _Dimensions> out)
{
	operate_omp([](_InType &in, _OutType &out) { out = in; }, in, out);
}

} // namespace ndim

#else // _OPENMP

#include "ndim/algorithm.h"

namespace ndim
{

template <typename _OperationType, size_t _Dimensions, typename... _Types>
void operate_omp(_OperationType op, std::tuple<pointer<_Types, _Dimensions>...> args)
{
	operate(op, args);
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
_Result accumulate_omp(_OperationType op, _Result value, std::tuple<pointer<_Types, _Dimensions>...> args)
{
	return accumulate(op, value, args);
}

template <typename _OperationType, size_t _Dimensions, typename... _Types>
void operate_omp(_OperationType op, pointer<_Types, _Dimensions>... args)
{
	operate(op, std::make_tuple(args...));
}

template <typename _Result, typename _OperationType, size_t _Dimensions, typename... _Types>
_Result accumulate_omp(_OperationType op, _Result value, pointer<_Types, _Dimensions>... args)
{
	return accumulate(op, value, std::make_tuple(args...));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform_omp(pointer<_OutType, _Dimensions> out, _OperationType op, std::tuple<pointer<_InTypes, _Dimensions>...> in)
{
	operate(_helper::makeAssignOperation(out, op, in), std::tuple_cat(std::make_tuple(out), in));
}

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
void transform_omp(pointer<_OutType, _Dimensions> out, _OperationType op, pointer<_InTypes, _Dimensions>... in)
{
	transform(out, op, std::make_tuple(in...));
}

template <typename _OutType, size_t _Dimensions, typename _InType>
void copy_omp(pointer<_InType, _Dimensions> in, pointer<_OutType, _Dimensions> out)
{
	operate([](_InType &in, _OutType &out) { out = in; }, in, out);
}

} // namespace ndim

#endif // _OPENMP

#endif // NDIM_ALGORITHM_OMP_H

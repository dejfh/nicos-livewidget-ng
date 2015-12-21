#ifndef _HELPER
#define _HELPER

#include <cstddef>
#include <tuple>
#include "ndim/pointer.h"

namespace ndim
{

namespace _helper
{

template <typename _OutType, typename _OperationType, typename... _InTypes>
struct AssignOperation {
	_OperationType op;
	AssignOperation(_OperationType op)
		: op(op)
	{
	}
	void operator()(_OutType &out, _InTypes &... in) const
	{
		out = op(in...);
	}
};

template <typename _OutType, size_t _Dimensions, typename _OperationType, typename... _InTypes>
AssignOperation<_OutType, _OperationType, _InTypes...> makeAssignOperation(pointer<_OutType, _Dimensions> &, _OperationType op, std::tuple<pointer<_InTypes, _Dimensions>...> &)
{
	return AssignOperation<_OutType, _OperationType, _InTypes...>(op);
}

} // namespace _helper

} // namespace ndim

#endif // _HELPER

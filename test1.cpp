#include <tuple>

#include "fc/datafilterbase.h"
#include "ndimfilter/transform.h"

template <typename Type>
void assert_lvalue(Type &&)
{
	static_assert(std::is_lvalue_reference<Type &&>::value, "no lvalue");
}
template <typename Type>
void assert_rvalue(Type &&)
{
	static_assert(std::is_rvalue_reference<Type &&>::value, "no rvalue");
}

void test1()
{
	using DataFilterType = const fc::DataFilter<float, 2>;

	using ElementOpType = std::minus<float>;
	using ElementOpResultType = std::result_of<ElementOpType(float, float)>::type;
	static_assert(std::is_same<ElementOpResultType, float>::value, "Wrong element type!");

	using DataOpType = fc::PerElementDataOperation<ElementOpType>;
	using DataOpResultType = std::result_of<DataOpType(fc::Container<float, 2>, fc::Container<float, 2>)>::type;
	static_assert(std::is_same<typename DataOpResultType::ElementType, float>::value, "Wrong data type!");
	static_assert(std::is_same<DataOpResultType, fc::Container<float, 2>>::value, "Wrong data type!");

	using FilterOpType = fc::DataFilterHandlerWithOperationBase<DataOpType, DataFilterType, DataFilterType>;
	using FilterType = fc::HandlerDataFilterBase<FilterOpType>;
	static_assert(FilterType::ResultDimensionality == 2, "Wrong filter dimensionality!");
	static_assert(std::is_same<FilterType::ResultElementType, float>::value, "Wrong filter type!");

	int a = 5;
	std::tuple<int, int &&, int &> b(a, std::move(a), a);

	using TUPLE = decltype(b);

	assert_lvalue(hlp::variadic::forwardFromTuple<0, TUPLE &>(b));
	assert_lvalue(hlp::variadic::forwardFromTuple<1, TUPLE &>(b));
	assert_lvalue(hlp::variadic::forwardFromTuple<2, TUPLE &>(b));

	assert_lvalue(hlp::variadic::forwardFromTuple<0>(b));
	assert_lvalue(hlp::variadic::forwardFromTuple<1>(b));
	assert_lvalue(hlp::variadic::forwardFromTuple<2>(b));

	assert_rvalue(hlp::variadic::forwardFromTuple<0, TUPLE>(b));
	assert_rvalue(hlp::variadic::forwardFromTuple<1, TUPLE>(b));
	assert_lvalue(hlp::variadic::forwardFromTuple<2, TUPLE>(b));

	assert_rvalue(hlp::variadic::forwardFromTuple<0>(std::move(b)));
	assert_rvalue(hlp::variadic::forwardFromTuple<1>(std::move(b)));
	assert_lvalue(hlp::variadic::forwardFromTuple<2>(std::move(b)));

	//	FilterType *ptr = new FilterType();
	//	const filter::DataFilter<float, 2> *filterPtr = ptr;
	//	hlp::unused(filterPtr);

	auto ptr = std::make_shared<FilterType>();
	std::shared_ptr<const fc::DataFilter<float, 2>> ptr2 = ptr;
	hlp::unused(ptr2);

	//	ElementOpType elementOp;
	//	DataOpType dataOp(elementOp);
	//	FilterOpType filterOp(dataOp, "blubb");
	//	auto filter = std::make_shared<FilterType>();
	//	filter->setOperation(filterOp);

	//	static_assert(std::is_same<FilterOpType::ResultContainerType, filter::Container<float, 2>>::value, "Wrong type!");

	//	static_assert(FilterOpType::ResultDimensionality == 2, "Wrong dimensionality!");
	//	DataOpType a = FilterOpType::ResultElementType(0);
	//	hlp::unused(a);
}

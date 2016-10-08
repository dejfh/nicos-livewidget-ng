struct FixDimFilter1d : public Filter1d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include "fc/filter/fixdim.h"
	using FixDimFilter1d = pyfc::InheritPtr<fc::filter::FixDim<float, 1>, pyfc::FilterNd<float, 1>>;
%End
private:
FixDimFilter1d(const FixDimFilter1d& ); // Disable allocation as array
public:
FixDimFilter1d();
%MethodCode
	sipCpp = FixDimFilter1d::create(std::make_shared<fc::filter::FixDim<float, 1>>());
%End
FixDimFilter1d(const FilterVar &a0);
%MethodCode
	sipCpp = FixDimFilter1d::create(std::make_shared<fc::filter::FixDim<float, 1>>());
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const FilterVar &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};

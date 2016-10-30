struct FixDimFilter1d : public Filter1d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include "fc/filter/fixdim.h"
	using FixDimFilter1d = pyfc::FinalPtr<fc::filter::FixDim<float, 1>, Filter1d>;
%End
public:
FixDimFilter1d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter1d>();
%End
FixDimFilter1d(const FilterVar &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter1d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const FilterVar &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};

struct FixDimFilter2d : public Filter2d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include "fc/filter/fixdim.h"
	using FixDimFilter2d = pyfc::FinalPtr<fc::filter::FixDim<float, 2>, Filter2d>;
%End
public:
FixDimFilter2d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter2d>();
%End
FixDimFilter2d(const FilterVar &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter2d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const FilterVar &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};

struct FixDimFilter3d : public Filter3d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include "fc/filter/fixdim.h"
	using FixDimFilter3d = pyfc::FinalPtr<fc::filter::FixDim<float, 3>, Filter3d>;
%End
public:
FixDimFilter3d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter3d>();
%End
FixDimFilter3d(const FilterVar &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter3d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const FilterVar &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};

struct FixDimFilter4d : public Filter4d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include "fc/filter/fixdim.h"
	using FixDimFilter4d = pyfc::FinalPtr<fc::filter::FixDim<float, 4>, Filter4d>;
%End
public:
FixDimFilter4d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter4d>();
%End
FixDimFilter4d(const FilterVar &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<FixDimFilter4d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const FilterVar &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};

class FixDimFilter1d
{
%TypeHeaderCode
#include "fc/filter/fixdim.h"
using FixDimFilter1d = std::shared_ptr<fc::filter::FixDim<float, 1>>;
%End
public:
	FixDimFilter1d();
	%MethodCode
		sipCpp = new FixDimFilter1d(std::make_shared<fc::filter::FixDim<float, 1>>());
	%End
	FixDimFilter1d(const FilterVar &a0);
	%MethodCode
		sipCpp = new FixDimFilter1d(std::make_shared<fc::filter::FixDim<float, 1>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const FilterVar &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End

	Filter1d filter() const;
	%MethodCode
		sipRes = new Filter1d(*sipCpp);
	%End
};

class FixDimFilter2d
{
%TypeHeaderCode
#include "fc/filter/fixdim.h"
using FixDimFilter2d = std::shared_ptr<fc::filter::FixDim<float, 2>>;
%End
public:
	FixDimFilter2d();
	%MethodCode
		sipCpp = new FixDimFilter2d(std::make_shared<fc::filter::FixDim<float, 2>>());
	%End
	FixDimFilter2d(const FilterVar &a0);
	%MethodCode
		sipCpp = new FixDimFilter2d(std::make_shared<fc::filter::FixDim<float, 2>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const FilterVar &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End

	Filter2d filter() const;
	%MethodCode
		sipRes = new Filter2d(*sipCpp);
	%End
};

class FixDimFilter3d
{
%TypeHeaderCode
#include "fc/filter/fixdim.h"
using FixDimFilter3d = std::shared_ptr<fc::filter::FixDim<float, 3>>;
%End
public:
	FixDimFilter3d();
	%MethodCode
		sipCpp = new FixDimFilter3d(std::make_shared<fc::filter::FixDim<float, 3>>());
	%End
	FixDimFilter3d(const FilterVar &a0);
	%MethodCode
		sipCpp = new FixDimFilter3d(std::make_shared<fc::filter::FixDim<float, 3>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const FilterVar &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End

	Filter3d filter() const;
	%MethodCode
		sipRes = new Filter3d(*sipCpp);
	%End
};

class FixDimFilter4d
{
%TypeHeaderCode
#include "fc/filter/fixdim.h"
using FixDimFilter4d = std::shared_ptr<fc::filter::FixDim<float, 4>>;
%End
public:
	FixDimFilter4d();
	%MethodCode
		sipCpp = new FixDimFilter4d(std::make_shared<fc::filter::FixDim<float, 4>>());
	%End
	FixDimFilter4d(const FilterVar &a0);
	%MethodCode
		sipCpp = new FixDimFilter4d(std::make_shared<fc::filter::FixDim<float, 4>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const FilterVar &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End

	Filter4d filter() const;
	%MethodCode
		sipRes = new Filter4d(*sipCpp);
	%End
};

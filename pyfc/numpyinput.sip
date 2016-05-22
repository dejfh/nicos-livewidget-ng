class Numpy1d
{
%TypeHeaderCode
#include "numpyinput.h"
using Numpy1d = std::shared_ptr<pyfc::NumpyInput<1>>;
%End
public:
	Numpy1d();
	%MethodCode
		sipCpp = new Numpy1d(std::make_shared<pyfc::NumpyInput<1>>());
	%End
	void setData(PyObject*);
	%MethodCode
		hlp::python::Ref obj(a0, true);
		(*sipCpp)->setData(std::move(obj));
	%End

	Filter1d filter() const;
	%MethodCode
		sipRes = new Filter1d(*sipCpp);
	%End
};

class Numpy2d
{
%TypeHeaderCode
#include "numpyinput.h"
using Numpy2d = std::shared_ptr<pyfc::NumpyInput<2>>;
%End
public:
	Numpy2d();
	%MethodCode
		sipCpp = new Numpy2d(std::make_shared<pyfc::NumpyInput<2>>());
	%End
	void setData(PyObject*);
	%MethodCode
		hlp::python::Ref obj(a0, true);
		(*sipCpp)->setData(std::move(obj));
	%End

	Filter2d filter() const;
	%MethodCode
		sipRes = new Filter2d(*sipCpp);
	%End
};

class Numpy3d
{
%TypeHeaderCode
#include "numpyinput.h"
using Numpy3d = std::shared_ptr<pyfc::NumpyInput<3>>;
%End
public:
	Numpy3d();
	%MethodCode
		sipCpp = new Numpy3d(std::make_shared<pyfc::NumpyInput<3>>());
	%End
	void setData(PyObject*);
	%MethodCode
		hlp::python::Ref obj(a0, true);
		(*sipCpp)->setData(std::move(obj));
	%End

	Filter3d filter() const;
	%MethodCode
		sipRes = new Filter3d(*sipCpp);
	%End
};

class Numpy4d
{
%TypeHeaderCode
#include "numpyinput.h"
using Numpy4d = std::shared_ptr<pyfc::NumpyInput<4>>;
%End
public:
	Numpy4d();
	%MethodCode
		sipCpp = new Numpy4d(std::make_shared<pyfc::NumpyInput<4>>());
	%End
	void setData(PyObject*);
	%MethodCode
		hlp::python::Ref obj(a0, true);
		(*sipCpp)->setData(std::move(obj));
	%End

	Filter4d filter() const;
	%MethodCode
		sipRes = new Filter4d(*sipCpp);
	%End
};

struct Numpy1d : Filter1d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy1d = pyfc::FinalPtr<pyfc::NumpyInput<1>, Filter1d>;
%End
Numpy1d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Numpy1d>();
%End
void setData(PyObject*);
	%MethodCode
		(*sipCpp)->setData(a0);
	%End
};

struct Numpy2d : Filter2d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy2d = pyfc::FinalPtr<pyfc::NumpyInput<2>, Filter2d>;
%End
Numpy2d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Numpy2d>();
%End
void setData(PyObject*);
%MethodCode
	(*sipCpp)->setData(a0);
%End
};

struct Numpy3d : Filter3d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy3d = pyfc::FinalPtr<pyfc::NumpyInput<3>, Filter3d>;
%End
Numpy3d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Numpy3d>();
%End
void setData(PyObject*);
	%MethodCode
		(*sipCpp)->setData(a0);
	%End
};

struct Numpy4d : Filter4d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy4d = pyfc::FinalPtr<pyfc::NumpyInput<4>, Filter4d>;
%End
Numpy4d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Numpy4d>();
%End
void setData(PyObject*);
	%MethodCode
		(*sipCpp)->setData(a0);
	%End
};

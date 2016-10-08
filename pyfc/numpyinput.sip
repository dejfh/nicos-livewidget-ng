struct Numpy1d : Filter1d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy1d = pyfc::FinalPtr<pyfc::NumpyInput<1>, pyfc::FilterNd<float, 1>>;
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
};

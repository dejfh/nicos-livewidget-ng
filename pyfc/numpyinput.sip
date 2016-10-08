struct Numpy1d : Filter1d
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/numpyinput.h>
using Numpy1d = pyfc::InheritPtr<pyfc::NumpyInput<1>, pyfc::FilterNd<float, 1>>;
%End
private:
Numpy1d(const Numpy1d& ); // Disable allocation as array
public:
Numpy1d();
%MethodCode
	sipCpp = Numpy1d::create(std::make_shared<pyfc::NumpyInput<1>>());
%End
void setData(PyObject*);
%MethodCode
	hlp::python::Ref obj(a0, true);
	(*sipCpp)->setData(std::move(obj));
%End
};

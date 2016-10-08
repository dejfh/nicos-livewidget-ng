struct InvokeFilter : FilterVar
{
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <pyfc/invokefilter.h>
using InvokeFilter = pyfc::InheritPtr<pyfc::InvokeFilter, pyfc::FilterVar<float>>;
%End
private:
InvokeFilter(const InvokeFilter& ); // Disable allocation as array
public:
InvokeFilter();
%MethodCode
	sipCpp = InvokeFilter::create(std::make_shared<pyfc::InvokeFilter>());
%End
void setPredecessors(PyObject *predecessors);
%MethodCode
	QVector<std::shared_ptr<const fc::DataFilterVar<float>>> predecessors;
	for (size_t i = 0, ii = PyList_GET_SIZE(a0); i < ii; ++i) {
		auto *predecessor = reinterpret_cast<std::shared_ptr<const fc::DataFilterVar<float>> *>(sipConvertToType(
			PyList_GET_ITEM(a0, i), sipType_FilterVar, 0, 0, 0, &sipIsErr));
		predecessors.append(*predecessor);
	}
	(*sipCpp)->setPredecessors(std::move(predecessors));
%End

void setTarget(PyObject *prepareProc, PyObject *getDataProc);
%MethodCode
   (*sipCpp)->setTarget(hlp::python::Ref(a0, true), hlp::python::Ref(a1, true));
%End
};

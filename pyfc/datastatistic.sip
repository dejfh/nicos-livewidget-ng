struct DataStatistic
{
%TypeHeaderCode
#include "ndimdata/statistic.h"
using DataStatistic = ndimdata::DataStatistic;
%End
public:
    double min();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->min);
    %End
    double max();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->max);
    %End
	void setMin(double);
	%MethodCode
		sipCpp->min = a0;
	%End
	void setMax(double);
	%MethodCode
		sipCpp->max = a0;
	%End

    long long count();
    %MethodCode
        return PyInt_FromSize_t(sipCpp->count);
    %End
    long long countNan();
    %MethodCode
        return PyInt_FromSize_t(sipCpp->count_nan);
    %End
	void setCount(long long);
	%MethodCode
		sipCpp->count = a0;
	%End
	void setCountNan(long long);
	%MethodCode
		sipCpp->count_nan = a0;
	%End

    double sum();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->sum);
    %End
    double sumSquares();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->sum_squares);
    %End
	void setSum(double);
	%MethodCode
		sipCpp->sum = a0;
	%End
	void setSumSquares(double);
	%MethodCode
		sipCpp->sum_squares = a0;
	%End

    double autoLowBound();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->auto_low_bound);
    %End
    double autoHighBound();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->auto_high_bound);
    %End
	void setAutoLowBound(double);
	%MethodCode
		sipCpp->auto_low_bound = a0;
	%End
	void setAutoHighBound(double);
	%MethodCode
		sipCpp->auto_high_bound = a0;
	%End

    double width() const;
    double mean_value() const;
    double variance() const;
    double standard_deviation() const;

    PyObject *histogram();
    %MethodCode
	  PyObject *list = PyList_New(sipCpp->histogram.size());
	  for (Py_ssize_t i = 0, ii = sipCpp->histogram.size(); i < ii; ++i) {
		PyList_SetItem(list, i, PyInt_FromSsize_t(sipCpp->histogram[i]));
	  }
	  sipRes = list;
    %End

	void setHistogram(PyObject* list);
	%MethodCode
		PyObject *list = a0;
		size_t size = PyList_Size(list);
		sipCpp->histogram.resize(size);
		for (Py_ssize_t i = 0, ii = size; i < ii; ++i) {
			Py_ssize_t value = PyInt_AsSsize_t(PyList_GetItem(list, i));
			sipCpp->histogram[i] = value;
		}
	%End
};

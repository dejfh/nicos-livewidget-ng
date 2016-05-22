struct DataStatistic /Abstract/
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

    long long count();
    %MethodCode
        return PyInt_FromSize_t(sipCpp->count);
    %End
    long long countNan();
    %MethodCode
        return PyInt_FromSize_t(sipCpp->count_nan);
    %End
    double sum();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->sum);
    %End
    double sumSquares();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->sum_squares);
    %End

    double autoLowBound();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->auto_low_bound);
    %End
    double autoHighBound();
    %MethodCode
        return PyFloat_FromDouble(sipCpp->auto_high_bound);
    %End

    double width() const;
    double mean_value() const;
    double variance() const;
    double standard_deviation() const;
};

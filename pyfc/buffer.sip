struct Validatable /Abstract/ {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
%End
bool isValid();
%MethodCode
	sipRes = PyBool_FromLong((*sipCpp)->isValid());
%End
};

struct Buffer1d : Filter1d, Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	using Buffer1d = pyfc::FinalPtr<fc::filter::Buffer<float, 1>, Filter1d, Validatable>;
%End
public:
Buffer1d();
%MethodCode
	sipCpp = new Buffer1d(std::make_shared<fc::filter::Buffer<float, 1>>());
%End
Buffer1d(const Filter1d &a0);
%MethodCode
	sipCpp = new Buffer1d(std::make_shared<fc::filter::Buffer<float, 1>>());
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const Filter1d &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
bool isValid();
%MethodCode
	sipRes = PyBool_FromLong((*sipCpp)->isValid());
%End
};

struct ImageBuffer : Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	#include <QImage>
	using ImageBuffer = pyfc::FinalPtr<fc::filter::Buffer<QImage>, Validatable>;
%End
private:
	ImageBuffer();
public:
bool isValid();
%MethodCode
	sipRes = PyBool_FromLong((*sipCpp)->isValid());
%End
QImage image();
%MethodCode
	if (!(*sipCpp)->isValid())
		sipRes = nullptr; // not validated.
	else if (!(*sipCpp)->data())
		sipRes = nullptr; // no data available.
	else
		sipRes = new QImage((*sipCpp)->data().first());
%End
};

class StatisticBuffer : Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	#include <ndimdata/statistic.h>
	using StatisticBuffer = pyfc::FinalPtr<fc::filter::Buffer<ndimdata::DataStatistic>, Validatable>;
%End
private:
	StatisticBuffer();
public:
bool isValid();
%MethodCode
	sipRes = PyBool_FromLong((*sipCpp)->isValid());
%End
DataStatistic statistic();
%MethodCode
	if (!(*sipCpp)->isValid())
		sipRes = nullptr; // not validated.
	else if (!(*sipCpp)->data())
		sipRes = nullptr; // no data available.
	else
		sipRes = new DataStatistic((*sipCpp)->data().first());
%End
};

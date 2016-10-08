struct Validatable {
%TypeHeaderCode
#include <memory>
#include <fc/filter.h>
using Validatable = std::shared_ptr<const fc::Validatable>;
%End
bool isValid();
%MethodCode
	sipRes = PyBool_FromLong((*sipCpp)->isValid());
%End
private:
	Validatable();
};

struct Buffer1d : Filter1d {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	using Buffer1d = pyfc::InheritPtr<fc::filter::Buffer<float, 1>, pyfc::FilterNd<float, 1>>;
%End
private:
Buffer1d(const Buffer1d& ); // Disable allocation as array
public:
Buffer1d();
%MethodCode
	sipCpp = Buffer1d::create(std::make_shared<fc::filter::Buffer<float, 1>>());
%End
Buffer1d(const Filter1d &a0);
%MethodCode
	sipCpp = Buffer1d::create(std::make_shared<fc::filter::Buffer<float, 1>>());
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
Validatable asValidatable();
%MethodCode
	std::shared_ptr<fc::filter::Buffer<float, 1>> abc = *sipCpp;
	sipRes = new std::shared_ptr<const fc::Validatable>(abc);
%End
};

struct ImageBuffer {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	#include <QImage>
	using ImageBuffer = std::shared_ptr<fc::filter::Buffer<QImage>>;
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
Validatable asValidatable();
%MethodCode
	sipRes = new Validatable(*sipCpp);
%End
};

class StatisticBuffer {
%TypeHeaderCode
	#include <memory>
	#include <fc/filter/buffer.h>
	#include <ndimdata/statistic.h>
	using StatisticBuffer = std::shared_ptr<fc::filter::Buffer<ndimdata::DataStatistic>>;
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
Validatable validatable() const;
%MethodCode
	sipRes = new Validatable(*sipCpp);
%End
};

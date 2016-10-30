struct Validatable /Abstract/ {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
%End
bool isValid();
%MethodCode
	sipRes = (*sipCpp)->isValid();
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
	sipCpp = pyfc::make_final_ptr<Buffer1d>();
%End
Buffer1d(const Filter1d &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer1d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const Filter1d &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};
struct Buffer2d : Filter2d, Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	using Buffer2d = pyfc::FinalPtr<fc::filter::Buffer<float, 2>, Filter2d, Validatable>;
%End
public:
Buffer2d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer2d>();
%End
Buffer2d(const Filter2d &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer2d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const Filter2d &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};
struct Buffer3d : Filter3d, Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	using Buffer3d = pyfc::FinalPtr<fc::filter::Buffer<float, 3>, Filter3d, Validatable>;
%End
public:
Buffer3d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer3d>();
%End
Buffer3d(const Filter3d &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer3d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const Filter3d &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
%End
};
struct Buffer4d : Filter4d, Validatable {
%TypeHeaderCode
	#include <pyfc/pyfilter.h>
	#include <fc/filter/buffer.h>
	using Buffer4d = pyfc::FinalPtr<fc::filter::Buffer<float, 4>, Filter4d, Validatable>;
%End
public:
Buffer4d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer4d>();
%End
Buffer4d(const Filter4d &a0);
%MethodCode
	sipCpp = pyfc::make_final_ptr<Buffer4d>();
	(*sipCpp)->setPredecessor(*a0);
%End
void setPredecessor(const Filter4d &a0);
%MethodCode
	(*sipCpp)->setPredecessor(*a0);
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
	sipRes = (*sipCpp)->isValid();
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

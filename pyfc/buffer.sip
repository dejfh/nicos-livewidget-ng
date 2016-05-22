class Buffer1d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/buffer.h>
using Buffer1d = std::shared_ptr<fc::filter::Buffer<float, 1>>;
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
		sipRes = (*sipCpp)->isValid();
	%End
};

class Buffer2d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/buffer.h>
using Buffer2d = std::shared_ptr<fc::filter::Buffer<float, 2>>;
%End
public:
	Buffer2d();
	%MethodCode
		sipCpp = new Buffer2d(std::make_shared<fc::filter::Buffer<float, 2>>());
	%End
	Buffer2d(const Filter2d &a0);
	%MethodCode
		sipCpp = new Buffer2d(std::make_shared<fc::filter::Buffer<float, 2>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const Filter2d &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End
	bool isValid();
	%MethodCode
		sipRes = (*sipCpp)->isValid();
	%End
};

class Buffer3d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/buffer.h>
using Buffer3d = std::shared_ptr<fc::filter::Buffer<float, 3>>;
%End
public:
	Buffer3d();
	%MethodCode
		sipCpp = new Buffer3d(std::make_shared<fc::filter::Buffer<float, 3>>());
	%End
	Buffer3d(const Filter3d &a0);
	%MethodCode
		sipCpp = new Buffer3d(std::make_shared<fc::filter::Buffer<float, 3>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const Filter3d &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End
	bool isValid();
	%MethodCode
		sipRes = (*sipCpp)->isValid();
	%End
};

class Buffer4d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/buffer.h>
using Buffer4d = std::shared_ptr<fc::filter::Buffer<float, 4>>;
%End
public:
	Buffer4d();
	%MethodCode
		sipCpp = new Buffer4d(std::make_shared<fc::filter::Buffer<float, 4>>());
	%End
	Buffer4d(const Filter4d &a0);
	%MethodCode
		sipCpp = new Buffer4d(std::make_shared<fc::filter::Buffer<float, 4>>());
		(*sipCpp)->setPredecessor(*a0);
	%End
	void setPredecessor(const Filter4d &a0);
	%MethodCode
		(*sipCpp)->setPredecessor(*a0);
	%End
	bool isValid();
	%MethodCode
		sipRes = (*sipCpp)->isValid();
	%End
};

class ImageBuffer {
%TypeHeaderCode
#include <memory>
#include <fc/filter/buffer.h>
#include <QImage>
using ImageBuffer = std::shared_ptr<fc::filter::Buffer<QImage>>;
%End
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
	Validatable validatable() const;
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

class Validatable {
%TypeHeaderCode
#include <memory>
#include <fc/filter.h>
using Validatable = std::shared_ptr<const fc::Validatable>;
%End

public:
	Validatable();
	Validatable(const Buffer1d &);
	Validatable(const Buffer2d &);
	Validatable(const Buffer3d &);
	Validatable(const Buffer4d &);
	Validatable(const ImageBuffer &);
	Validatable(const StatisticBuffer &);
};

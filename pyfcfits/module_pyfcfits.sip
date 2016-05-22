
%Module(name=nicos_filterchain_fits, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%Import ../pyfc/module_pyfc.sip

class Fits1d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/fits.h>
using Filter1d = std::shared_ptr<const fc::DataFilter<float, 1>>;
using Fits1d = std::shared_ptr<fc::filter::fits::Loader<float, 1>>;
%End
public:
	Fits1d();
	%MethodCode
		sipCpp = new Fits1d(std::make_shared<fc::filter::fits::Loader<float, 1>>());
	%End
	void setFilename(const QString &a0);
	%MethodCode
		(*sipCpp)->setFilename(*a0);
	%End

	Filter1d filter() const;
	%MethodCode
		sipRes = new Filter1d(*sipCpp);
	%End
};

class Fits2d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/fits.h>
using Filter2d = std::shared_ptr<const fc::DataFilter<float, 2>>;
using Fits2d = std::shared_ptr<fc::filter::fits::Loader<float, 2>>;
%End
public:
	Fits2d();
	%MethodCode
		sipCpp = new Fits2d(std::make_shared<fc::filter::fits::Loader<float, 2>>());
	%End
	void setFilename(const QString &a0);
	%MethodCode
		(*sipCpp)->setFilename(*a0);
	%End

	Filter2d filter() const;
	%MethodCode
		sipRes = new Filter2d(*sipCpp);
	%End
};

class Fits3d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/fits.h>
using Filter3d = std::shared_ptr<const fc::DataFilter<float, 3>>;
using Fits3d = std::shared_ptr<fc::filter::fits::Loader<float, 3>>;
%End
public:
	Fits3d();
	%MethodCode
		sipCpp = new Fits3d(std::make_shared<fc::filter::fits::Loader<float, 3>>());
	%End
	void setFilename(const QString &a0);
	%MethodCode
		(*sipCpp)->setFilename(*a0);
	%End

	Filter3d filter() const;
	%MethodCode
		sipRes = new Filter3d(*sipCpp);
	%End
};

class Fits4d {
%TypeHeaderCode
#include <memory>
#include <fc/filter/fits.h>
using Filter4d = std::shared_ptr<const fc::DataFilter<float, 4>>;
using Fits4d = std::shared_ptr<fc::filter::fits::Loader<float, 4>>;
%End
public:
	Fits4d();
	%MethodCode
		sipCpp = new Fits4d(std::make_shared<fc::filter::fits::Loader<float, 4>>());
	%End
	void setFilename(const QString &a0);
	%MethodCode
		(*sipCpp)->setFilename(*a0);
	%End

	Filter4d filter() const;
	%MethodCode
		sipRes = new Filter4d(*sipCpp);
	%End
};

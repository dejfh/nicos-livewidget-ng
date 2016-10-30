
%Module(name=nicos_filterchain_fits, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%Import ../pyfc/module_pyfc.sip

struct Fits1d : Filter1d {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <fc/filter/fits.h>
using Fits1d = pyfc::FinalPtr<fc::filter::fits::Loader<float, 1>, Filter1d>;
%End
Fits1d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Fits1d>();
%End
void setFilename(const QString &a0);
%MethodCode
	(*sipCpp)->setFilename(*a0);
%End
};

struct Fits2d : Filter2d {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <fc/filter/fits.h>
using Fits2d = pyfc::FinalPtr<fc::filter::fits::Loader<float, 2>, Filter2d>;
%End
Fits2d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Fits2d>();
%End
void setFilename(const QString &a0);
%MethodCode
	(*sipCpp)->setFilename(*a0);
%End
};

struct Fits3d : Filter3d {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <fc/filter/fits.h>
using Fits3d = pyfc::FinalPtr<fc::filter::fits::Loader<float, 3>, Filter3d>;
%End
Fits3d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Fits3d>();
%End
void setFilename(const QString &a0);
%MethodCode
	(*sipCpp)->setFilename(*a0);
%End
};

struct Fits4d : Filter4d {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <fc/filter/fits.h>
using Fits4d = pyfc::FinalPtr<fc::filter::fits::Loader<float, 4>, Filter4d>;
%End
Fits4d();
%MethodCode
	sipCpp = pyfc::make_final_ptr<Fits4d>();
%End
void setFilename(const QString &a0);
%MethodCode
	(*sipCpp)->setFilename(*a0);
%End
};

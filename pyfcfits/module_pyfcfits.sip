
%Module(name=nicos_filterchain_fits, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%Import ../pyfc/module_pyfc.sip

struct Fits1d : Filter1d {
%TypeHeaderCode
#include <pyfc/pyfilter.h>
#include <fc/filter/fits.h>
using Fits1d = pyfc::InheritPtr<fc::filter::fits::Loader<float, 1>, pyfc::FilterNd<float, 1>>;
%End
private:
Fits1d(const Fits1d& ); // Disable allocation as array
public:
Fits1d();
%MethodCode
	sipCpp = Fits1d::create(std::make_shared<fc::filter::fits::Loader<float, 1>>());
%End
void setFilename(const QString &a0);
%MethodCode
	(*sipCpp)->setFilename(*a0);
%End
};

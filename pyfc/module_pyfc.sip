
%Module(name=nicos_filterchain, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%ModuleCode
void init_pyfc();
%End

%InitialisationCode
init_pyfc();
%End

%Include filter.sip
%Include fixdimfilter.sip
%Include buffer.sip
%Include datastatistic.sip
%Include invokefilter.sip
%Include imageoutput.sip
%Include numpyinput.sip
%Include validator.sip

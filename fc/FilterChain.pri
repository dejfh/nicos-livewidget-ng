 

HEADERS += \
    $$PWD/filter.h \
    $$PWD/filterbase.h \
    $$PWD/validation/validator.h \
    $$PWD/validation/watcher.h \
    $$PWD/gethelper.h \
    $$PWD/ndimcontainer.h \
    $$PWD/chains/fitspile.h \
    $$PWD/chains/darkimageopenbeam.h \
    $$PWD/chains/pixmapoutput.h \
    $$PWD/chains/profileplot.h \
    $$PWD/chains/zplot.h \
    $$PWD/chains/sourceselect.h \
    $$PWD/chains/dataprocess.h \
    $$PWD/datafilterbase.h \
    $$PWD/filter/perelement.h \
    $$PWD/filter/buffer.h \
    $$PWD/filter/switch.h \
    $$PWD/filter/accumulate.h \
    $$PWD/filter/extend.h \
    $$PWD/filter/fits.h \
    $$PWD/datafilter.h \
    $$PWD/filter/median.h \
    $$PWD/filter/analyzer.h \
    $$PWD/filter/pile.h \
    $$PWD/filter/profile.h \
    $$PWD/filter/subrange.h \
    $$PWD/filter/pixmap.h \
    $$PWD/filter/valuerange.h \
    $$PWD/filter/mapmerge.h \
    $$PWD/filter/input.h

SOURCES += \
    $$PWD/validation/validator.cpp \
    $$PWD/validation/watcher.cpp \
    $$PWD/chains/darkimageopenbeam.cpp \
    $$PWD/test/filterheader.cpp \
    $$PWD/test/filterinstance.cpp \
    $$PWD/test/filterinstanceauto.cpp

DISTFILES += \
    $$PWD/filter.dox

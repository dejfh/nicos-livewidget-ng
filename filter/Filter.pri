 

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
    $$PWD/fs/perelement.h \
    $$PWD/fs/buffer.h \
    $$PWD/fs/switch.h \
    $$PWD/fs/accumulate.h \
    $$PWD/fs/extend.h \
    $$PWD/fs/fits.h \
    $$PWD/datafilter.h \
    $$PWD/fs/median.h \
    $$PWD/fs/analyzer.h \
    $$PWD/fs/pile.h \
    $$PWD/fs/profile.h \
    $$PWD/fs/subrange.h \
    $$PWD/fs/pixmap.h \
    $$PWD/fs/valuerange.h \
    $$PWD/fs/mapmerge.h \
    $$PWD/fs/input.h

SOURCES += \
    $$PWD/validation/validator.cpp \
    $$PWD/validation/watcher.cpp \
    $$PWD/chains/darkimageopenbeam.cpp \
    $$PWD/test/filterheader.cpp \
    $$PWD/test/filterinstance.cpp \
    $$PWD/test/filterinstanceauto.cpp

DISTFILES += \
    $$PWD/filter.dox

 

HEADERS += \
    $$PWD/buffer.h \
    $$PWD/filter.h \
    filter/input.h \
    filter/switch.h \
    $$PWD/filterbase.h \
    $$PWD/validation/validator.h \
    $$PWD/mapmerge.h \
    $$PWD/validation/watcher.h \
    $$PWD/gethelper.h \
    $$PWD/ndimcontainer.h \
    $$PWD/typetraits.h \
    $$PWD/chains/fitspile.h \
    $$PWD/chains/darkimageopenbeam.h \
    $$PWD/chains/pixmapoutput.h \
    $$PWD/chains/profileplot.h \
    $$PWD/chains/zplot.h \
    $$PWD/chains/sourceselect.h \
    $$PWD/chains/dataprocess.h

SOURCES += \
    $$PWD/validation/validator.cpp \
    $$PWD/validation/watcher.cpp

DISTFILES += \
    $$PWD/filter.dox

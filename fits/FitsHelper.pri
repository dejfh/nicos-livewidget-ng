HEADERS += \
    $$PWD/fitsloader.h \
    $$PWD/fitsloadersettings.h

SOURCES += \
    $$PWD/fitsloader.cpp \
    $$PWD/fitsloadersettings.cpp

CONFIG *= cfitsio

DISTFILES += \
    $$PWD/fitshelper.dox

CONFIG += qwt

HEADERS += \
    $$PWD/geometryhelper.h \
    $$PWD/fixedratiotransform.h \
    $$PWD/griditem.h \
    $$PWD/histogramplot.h \
    $$PWD/plot2dtransform.h \
    $$PWD/rangeselectwidget.h \
    $$PWD/selectionlineitem.h \
    $$PWD/selectionrectitem.h \
    $$PWD/trackeritem.h \
    $$PWD/imageplot.h \
    $$PWD/handleitem.h

SOURCES += \
    $$PWD/fixedratiotransform.cpp \
    $$PWD/geometryhelper.cpp \
    $$PWD/griditem.cpp \
    $$PWD/histogramplot.cpp \
    $$PWD/rangeselectwidget.cpp \
    $$PWD/selectionlineitem.cpp \
    $$PWD/selectionrectitem.cpp \
    $$PWD/trackeritem.cpp \
    $$PWD/imageplot.cpp \
    $$PWD/handleitem.cpp

FORMS += \
    $$PWD/rangeselectwidget.ui \
    $$PWD/imageplot.ui

win32 {
    CONFIG(debug, debug|release) {
	LIBS *= -lqwtd
    } else {
	LIBS *= -lqwt
    }
}
unix {
    LIBS *= -lqwt
}

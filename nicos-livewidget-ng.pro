TEMPLATE = subdirs

SUBDIRS += \
    fc \
    tomo \
    fctest \
    tomotest \
    tomorun \
    pyfc \
    pyipw \
    pydemo \
    pyfctest \
    pytomo

fctest.depends = fc
pyfc.depends = fc
pydemo.depends = pyfc pyipw
tomotest.depends = tomo
tomorun.depends = tomo
pyfctest.depends = pyfc
pytomo.depends = tomo

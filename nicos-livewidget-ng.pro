TEMPLATE = subdirs

SUBDIRS += \
    fc \
    tomo \
    fctest \
    tomotest \
    tomorun \
    pyfc \
    pyfcfits \
    pyipw \
    pydemo \
    pyfctest

fctest.depends = fc
pyfc.depends = fc
pydemo.depends = pyfc pyipw
tomotest.depends = tomo
tomorun.depends = tomo
pyfctest.depends = pyfc
pytomo.depends = tomo

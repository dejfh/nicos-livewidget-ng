TEMPLATE = subdirs

SUBDIRS += \
    fc \
    tomo \
    fctest \
    tomotest \
    pyfc \
    pyipw \
    pydemo \
    pyfctest

fctest.depends = fc
pyfc.depends = fc
pydemo.depends = pyfc pyipw
tomotest.depends = fc
pyfctest.depends = pyfc

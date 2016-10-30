TEMPLATE = subdirs

SUBDIRS += \
    fc \
    pyfc \
    pyfcfits \
    pyipw \
    tomo \
    pytomo \
    pydemo

pyfc.depends = fc
pyipw.depends = pyfc pyfcfits
pytomo.depends = tomo
pydemo.depends = pyfc pyfcfits pyipw pytomo

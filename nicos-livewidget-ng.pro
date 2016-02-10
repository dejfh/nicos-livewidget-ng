TEMPLATE = subdirs

SUBDIRS += \
    pyfc \
    pyipw \
    fctest \
    pydemo

pydemo.depends = pyfc pyipw

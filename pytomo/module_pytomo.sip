
%Module(name=nicos_tomography, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%ModuleCode
void init_tomo();
%End

%InitialisationCode
init_tomo();
%End

class Tomography : public QObject
{
%TypeHeaderCode
#include <pytomo/tomography.h>
%End
private:
    Tomography(const Tomography&);
public:
    explicit Tomography(long long sinogramResolution, long long maxAngleCount, float center);
    ~Tomography();

signals:
    void stepDone();

public slots:
    void setOpenBeam(PyObject *openBeam);
    void appendSinogram(PyObject *sinogram, PyObject *angles);
    void setReconstruction(PyObject *reconstruction);

    void run(int stepCount);
    void stop();
    bool running();

    void setForceSteps(bool forceSteps);
    bool forceSteps();

    void requestReconstruction();
    bool reconstructionAvailable() const;
    PyObject *getReconstruction();
    void requestSinogram();
    bool sinogramAvailable() const;
    PyObject *getSinogram();
    void requestLikelihood();
    bool likelihoodAvailable() const;
    PyObject *getLikelihood();
    void requestGradient();
    bool gradientAvailable() const;
    PyObject *getGradient();
};

#ifndef TOMO_RECONSTRUCTOR_H
#define TOMO_RECONSTRUCTOR_H

#include "tomo/reconstructorerrors.h"

#include <memory>

#include <vector>
#include <array>

#include "ndim/pointer.h"

#include "helper/fixedpoint.h"

#include <QScopedPointer>

class QGLContext;
class QThread;

namespace tomo
{

class Reconstructor
{
	class ReconstructorPrivate;

public:
	enum ReconTextures {
		TexIntensity,
		TexSinoOrg,
		TexAngleMatrix,
		TexReconTest,
		TexSinoTest,
		TexLikelihood,
		TexSum,
		TexSinoRecon,
		TexRecon,
		TexGradient,
		TexCount
	};

private:
	QScopedPointer<ReconstructorPrivate> m;

	void acceptTestDirect();

public:
	explicit Reconstructor(const QGLContext *glContext);
	~Reconstructor();

	void reset();

	int sinogramResolution() const;
	int reconstructionResolution() const;

	void prepare(int resolution, int sinogram_capacity, float center);
	void setOpenBeam(ndim::pointer<const float, 1> data);
	void setSinogram(ndim::pointer<const float, 2> data, ndim::pointer<const float, 1> angles);
	void appendSinogram(ndim::pointer<const float, 2> data, ndim::pointer<const float, 1> angles);
	void setReconstruction(ndim::pointer<const float, 2> data);

	void guess();
	bool step();
	void clear();

	void readTexture(ndim::pointer<quint16, 2> data, ReconTextures texture);
	void readTexture(ndim::pointer<float, 2> data, ReconTextures texture);
	void readReconstruction(ndim::pointer<float, 2> data);

	float stepSize() const;
	void setStepSize(float value);
};

} // namespace tomo

#endif // TOMO_RECONSTRUCTOR_H

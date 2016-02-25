#ifndef TOMO_SINOGRAMFILE_H
#define TOMO_SINOGRAMFILE_H

#include <QObject>
#include <QFile>
#include <QString>
#include "ndim/pointer.h"

#include <QVector>
#include <valarray>

#include <QScopedPointer>

#include "tomo/axisofrotation.h"
#include "helper/fixedpoint.h"

namespace tomo
{

struct SinogramFileHeader;

class SinogramFile : public QObject
{
	Q_OBJECT

	mutable QFile m_file;

private:
	uchar *m_map_read;
	uchar *m_map_write;

	const SinogramFileHeader *header() const;

public:
	~SinogramFile();
	explicit SinogramFile(const QString &filename, QObject *parent = 0);
	explicit SinogramFile(const QString &filename, quint32 resolution, quint32 layers, quint32 angles, quint32 angle_capacity = 0, QObject *parent = 0);

	size_t resolution() const;
	size_t layers() const;
	size_t angleCount() const;
	void setAngleCount(size_t count);
	size_t capacity() const;

	float center(size_t layer) const;
	AxisOfRotation axis() const;
	void setAxis(const AxisOfRotation &axis);

	//	QVector<float> angles() const;
	//	void setAngles(const QVector<float> &angles);

	ndim::pointer<const hlp::FixedPoint<0x10000>, 1> mapAngles() const;
	// coordinates: x, index, layer
	ndim::pointer<const quint16, 3> mapDarkImage() const;
	// coordinates: x, index, layer
	ndim::pointer<const quint16, 3> mapOpenBeam() const;
	// coordinates: x, angle, layer
	ndim::pointer<const quint16, 3> mapSinogram() const;

	ndim::pointer<hlp::FixedPoint<0x10000>, 1> mapAnglesWrite();
	// coordinates: x, index, layer
	ndim::pointer<quint16, 3> mapDarkImageWrite();
	// coordinates: x, index, layer
	ndim::pointer<quint16, 3> mapOpenBeamWrite();
	// coordinates: x, angle, layer
	ndim::pointer<quint16, 3> mapSinogramWrite();

	void mapWrites();
	void unmapWrites();

signals:
	void angleAppended();
};

} // namespace tomo

#endif // TOMO_SINOGRAMFILE_H

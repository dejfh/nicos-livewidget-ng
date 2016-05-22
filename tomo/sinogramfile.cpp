#include "tomo/sinogramfile.h"

#include "tomo/sinogramfileheader.h"

#include <QDataStream>

#include "helper/helper.h"

#include "ndim/algorithm.h"

using hlp::assert_result;
using hlp::cast_over_void;

namespace tomo
{

SinogramFile::~SinogramFile()
{
}

SinogramFile::SinogramFile(const QString &filename, QObject *parent)
	: QObject(parent)
	, m_file(filename)
	, m_map_read(0)
	, m_map_write(0)
{
	assert_result(m_file.open(QFile::ReadWrite));
	assert_result((m_map_read = m_file.map(0, m_file.size())));

	const SinogramFileHeader *header = cast_over_void<const SinogramFileHeader *>(m_map_read);
	assert(header->validate(m_file.size()));
	hlp::unused(header);
}

SinogramFile::SinogramFile(const QString &filename, quint32 resolution, quint32 layers, quint32 angles, quint32 angle_capacity, QObject *parent)
	: QObject(parent)
	, m_file(filename)
	, m_map_read(0)
	, m_map_write(0)
{
	SinogramFileHeader header;
	header.build(resolution, layers, angles, angle_capacity);

	assert_result(m_file.open(QFile::ReadWrite | QFile::Truncate));
	assert_result(m_file.resize(header.total_size()));
	assert_result((m_map_read = m_file.map(0, m_file.size())));

	mapWrites();
	SinogramFileHeader *p_header = cast_over_void<SinogramFileHeader *>(m_map_write);
	*p_header = header;
	assert(p_header->validate(m_file.size()));
}

const SinogramFileHeader *SinogramFile::header() const
{
	return cast_over_void<const SinogramFileHeader *>(m_map_read);
}

size_t SinogramFile::resolution() const
{
	return header()->resolution;
}

size_t SinogramFile::layers() const
{
	return header()->layer_count;
}

size_t SinogramFile::angleCount() const
{
	return header()->angle_count;
}

void SinogramFile::setAngleCount(size_t count)
{
	mapWrites();
	SinogramFileHeader *header = cast_over_void<SinogramFileHeader *>(m_map_write);
	assert(count <= header->angle_capacity);
	header->angle_count = assert_cast<quint16>(count);
}

size_t SinogramFile::capacity() const
{
	return header()->angle_capacity;
}

float SinogramFile::center(size_t layer) const
{
	auto axis = AxisOfRotation::fromEdge(header()->resolution, header()->layer_count, header()->center, header()->tan_of_tilt);
	return axis.axisXOfLayer(layer);
}

AxisOfRotation SinogramFile::axis() const
{
	return AxisOfRotation::fromEdge(header()->resolution, header()->layer_count, header()->center, header()->tan_of_tilt);
}

void SinogramFile::setAxis(const AxisOfRotation &axis)
{
	mapWrites();
	SinogramFileHeader *header = cast_over_void<SinogramFileHeader *>(m_map_write);
	header->center = axis.center_layer_0;
	header->tan_of_tilt = axis.tan;
}

ndim::pointer<const hlp::FixedPoint<0x10000>, 1> SinogramFile::mapAngles() const
{
	return ndim::make_ptr_contiguous(cast_over_void<const hlp::FixedPoint<0x10000> *>(m_map_read + sizeof(SinogramFileHeader)), header()->angle_count);
}

// coordinates: x, index, layer
ndim::pointer<const quint16, 3> SinogramFile::mapDarkImage() const
{
	return ndim::make_ptr_contiguous(cast_over_void<const quint16 *>(m_map_read + header()->darkimage_offset), header()->resolution,
		header()->darkimage_count, header()->layer_count);
}

// coordinates: x, index, layer
ndim::pointer<const quint16, 3> SinogramFile::mapOpenBeam() const
{
	return ndim::make_ptr_contiguous(cast_over_void<const quint16 *>(m_map_read + header()->openbeam_offset), header()->resolution,
		header()->openbeam_count, header()->layer_count);
}

// coordinates: x, angle, layer
ndim::pointer<const quint16, 3> SinogramFile::mapSinogram() const
{
	ndim::pointer<const quint16, 3> sinograms = ndim::make_ptr_contiguous(cast_over_void<const quint16 *>(m_map_read + header()->sinogram_offset),
		header()->resolution, header()->angle_capacity, header()->layer_count);
	sinograms.selectRange(1, 0, header()->angle_count);
	return sinograms;
}

ndim::pointer<hlp::FixedPoint<0x10000>, 1> SinogramFile::mapAnglesWrite()
{
	mapWrites();
	return ndim::make_ptr_contiguous(cast_over_void<hlp::FixedPoint<0x10000> *>(m_map_read + sizeof(SinogramFileHeader)), header()->angle_capacity);
}

// coordinates: x, index, layer
ndim::pointer<quint16, 3> SinogramFile::mapDarkImageWrite()
{
	mapWrites();
	return ndim::make_ptr_contiguous(
		cast_over_void<quint16 *>(m_map_write + header()->darkimage_offset), header()->resolution, header()->darkimage_count, header()->layer_count);
}

// coordinates: x, index, layer
ndim::pointer<quint16, 3> SinogramFile::mapOpenBeamWrite()
{
	mapWrites();
	return ndim::make_ptr_contiguous(
		cast_over_void<quint16 *>(m_map_write + header()->openbeam_offset), header()->resolution, header()->openbeam_count, header()->layer_count);
}

// coordinates: x, angle, layer
ndim::pointer<quint16, 3> SinogramFile::mapSinogramWrite()
{
	mapWrites();
	ndim::pointer<quint16, 3> sinograms = ndim::make_ptr_contiguous(
		cast_over_void<quint16 *>(m_map_write + header()->sinogram_offset), header()->resolution, header()->angle_capacity, header()->layer_count);
	return sinograms;
}

void SinogramFile::mapWrites()
{
	if (!m_map_write)
		assert_result((m_map_write = m_file.map(0, m_file.size())));
}

void SinogramFile::unmapWrites()
{
	if (m_map_write) {
		m_file.unmap(m_map_write);
		m_map_write = 0;
	}
}

} // namespace tomo

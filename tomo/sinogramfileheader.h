#ifndef TOMO_SINOGRAMFILEHEADER_H
#define TOMO_SINOGRAMFILEHEADER_H

#include <QtCore>
#include <limits>

#include "helper/assertcast.h"
#include "helper/fixedpoint.h"

using hlp::assert_cast;

namespace tomo
{

struct SinogramFileHeader {
	static const quint32 normal_magic_number = 0x655f12c0;
	static const quint32 false_endian_magic_number = 0xc0125f65;
	static const quint32 current_version = 3;

	quint32 magic_number;
	quint32 version;
	quint16 resolution;
	quint16 layer_count;
	quint16 darkimage_count;
	quint16 openbeam_count;
	quint16 angle_count;
	quint16 angle_capacity;
	quint32 darkimage_offset;
	quint32 openbeam_offset;
	quint32 sinogram_offset;
	hlp::FixedPoint<0x10000> center;
	hlp::FixedPoint<0x10000> tan_of_tilt;

	size_t line_size() const
	{
		return resolution * sizeof(quint16);
	}
	size_t image_size() const
	{
		return layer_count * line_size();
	}
	size_t sinogram_size() const
	{
		return angle_capacity * line_size();
	}

	size_t darkimage_pos(size_t layer) const
	{
		return darkimage_offset + layer * darkimage_count;
	}
	size_t openbeam_pos(size_t layer) const
	{
		return openbeam_offset + layer * openbeam_count;
	}
	size_t sinogram_pos(size_t layer) const
	{
		return sinogram_offset + layer * sinogram_size();
	}

	size_t darkimage_line_pos(size_t layer, size_t index) const
	{
		return darkimage_pos(layer) + index * line_size();
	}
	size_t openbeam_line_pos(size_t layer, size_t index) const
	{
		return openbeam_pos(layer) + index * line_size();
	}
	size_t sinogram_line_pos(size_t layer, size_t index_angle) const
	{
		return sinogram_pos(layer) + index_angle * line_size();
	}

	size_t total_size() const
	{
		return sinogram_offset + layer_count * sinogram_size();
	}

	bool validate(size_t file_size = std::numeric_limits<size_t>::max()) const
	{
		return (magic_number == normal_magic_number)												//
			   && (version == current_version)														//
			   && (darkimage_offset >= sizeof(SinogramFileHeader) + angle_capacity * sizeof(float)) //
			   && (openbeam_offset >= darkimage_offset + darkimage_count * image_size())			//
			   && (sinogram_offset >= openbeam_offset + openbeam_count * image_size())				//
			   && (file_size >= total_size());
	}

	void calc_offsets()
	{
		size_t offset = sizeof(SinogramFileHeader);

		darkimage_offset = assert_cast<quint32>(offset += size_t(angle_capacity) * sizeof(float));
		openbeam_offset = assert_cast<quint32>(offset += darkimage_count * image_size());
		sinogram_offset = assert_cast<quint32>(offset += openbeam_count * image_size());
	}

	void build(size_t resolution, size_t layers, size_t angles, size_t angle_capacity)
	{
		this->magic_number = normal_magic_number;
		this->version = current_version;
		this->resolution = assert_cast<quint16>(resolution);
		this->layer_count = assert_cast<quint16>(layers);
		this->darkimage_count = 1;
		this->openbeam_count = 1;
		this->angle_count = assert_cast<quint16>(angles);
		this->angle_capacity = assert_cast<quint16>(std::max(angles, angle_capacity));
		calc_offsets();
	}
};

} // namespace tomo

#endif // TOMO_SINOGRAMFILEHEADER_H

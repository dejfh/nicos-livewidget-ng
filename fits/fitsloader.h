#ifndef FITS_FITSLOADER_H
#define FITS_FITSLOADER_H

#include <valarray>
#include <array>
#include <cassert>
#include <algorithm>

#include <QScopedPointer>
#include <QString>
#include <QMutex>
#include <QMutexLocker>
#include <QMap>
#include <QVariant>

#include <cstdint>

#include "helper/asyncprogress.h"

#include "ndim/pointer.h"
#include "ndim/range.h"
#include "ndim/iterator.h"
#include "ndim/buffer.h"
#include "ndim/algorithm_omp.h"

#include "helper/helper.h"

struct FitsLoaderSettings;

namespace fitshelper
{

enum ValueType {
	//! signed, 8bit
	ValueTypeChar,
	//! signed, 8bit (alias for ValueTypeChar)
	ValueTypeSChar = ValueTypeChar,
	//! signed, 16bit
	ValueTypeShort,
	//! signed, 32bit
	ValueTypeLong,
	//! signed, 64bit
	ValueTypeLongLong,
	//! unsigned, 8bit
	ValueTypeUChar,
	//! unsigned, 16bit
	ValueTypeUShort,
	//! unsigned, 32bit
	ValueTypeULong,
	//! float, 32 bit
	ValueTypeFloat,
	//! float, 64 bit
	ValueTypeDouble
};

inline ValueType getValueType(int8_t)
{
	return ValueTypeSChar;
}
inline ValueType getValueType(int16_t)
{
	return ValueTypeShort;
}
inline ValueType getValueType(int32_t)
{
	return ValueTypeLong;
}
inline ValueType getValueType(int64_t)
{
	return ValueTypeLongLong;
}
inline ValueType getValueType(uint8_t)
{
	return ValueTypeUChar;
}
inline ValueType getValueType(uint16_t)
{
	return ValueTypeUShort;
}
inline ValueType getValueType(uint32_t)
{
	return ValueTypeULong;
}
inline ValueType getValueType(float)
{
	return ValueTypeFloat;
}
inline ValueType getValueType(double)
{
	return ValueTypeDouble;
}
template <typename _T>
ValueType getValueType()
{
	return getValueType(_T());
}

class FitsHelper
{
	// cfitsio may not be reentrant.
	// In the future this may be checked with fits_is_reentrant();
	// TODO: Don't use mutex when cfitsio is reentrant.

	static QMutex *getMutex();

	QMutexLocker lock;

protected:
	void *fitsfile;
	const QString filename;

public:
	FitsHelper(const QString &filename);
	~FitsHelper();

protected:
	size_t _getDimensions() const;
	void _getSizes(size_t *sizes, size_t dimensions) const;
	void _readContiguous(ValueType valueType, void *data, size_t count) const;
	void _readContiguousSubimage(ValueType valueType, void *data, size_t *start, size_t *sizes, size_t dimensions) const;

	QString _readKey(const QString &name, QString *comment = 0) const;
	void _readKey(int index, QString &name, QString &value, QString *comment = 0, bool *isUserKey = 0) const;
	QMap<QString, QString> _readUserKeys(QMap<QString, QString> *comments = 0) const;

public:
	template <size_t _D>
	ndim::sizes<_D> dimensions()
	{
		QMutexLocker lock(getMutex());
		hlp::unused(lock);
		ndim::sizes<_D> dimensions;
		_getSizes(dimensions.data(), _D);
		return dimensions;
	}

	template <typename _T, size_t _D>
	void read(ndim::pointer<_T, _D> data)
	{
		QMutexLocker lock(getMutex());
		hlp::unused(lock);
		bool contiguous = data.isContiguous();
		ndim::Buffer<_T, _D> buffer;
		ndim::pointer<_T, _D> pointer = data;
		if (!contiguous) {
			buffer.resize(data.sizes);
			pointer = buffer.pointer();
		}
		_readContiguous(getValueType<_T>(), pointer.data, pointer.size());
		if (!contiguous) {
#pragma omp parallel
			{
				ndim::copy_omp(pointer, data);
			}
		}
	}

	QMap<QString, QString> readUserKeys(QMap<QString, QString> *comments = 0) const
	{
		QMutexLocker lock(getMutex());
		hlp::unused(lock);

		return _readUserKeys(comments);
	}

	template <typename _T, size_t _D>
	void read(ndim::pointer<_T, _D> data, ndim::range<_D> range)
	{
		QMutexLocker lock(getMutex());
		hlp::unused(lock);

		bool contiguous = data.isContiguous();
		ndim::Buffer<_T, _D> buffer;
		ndim::pointer<_T, _D> pointer = data;
		if (!contiguous) {
			buffer.resize(data.sizes);
			pointer = buffer.pointer();
		}
		_readContiguousSubimage(getValueType<_T>(), pointer.data, range.coords.data(), range.sizes.data(), _D);
		if (!contiguous) {
#pragma omp parallel
			{
				ndim::copy_omp(pointer, data);
			}
		}
	}
};

class TomoFitsHelper : public FitsHelper
{
protected:
	const QString filename;

public:
	TomoFitsHelper(const QString &filename);

	double getAngle(const FitsLoaderSettings &settings) const;
};

} // namespace fitshelper

#endif // FITS_FITSLOADER_H

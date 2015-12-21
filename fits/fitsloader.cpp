#include "fits/fitsloader.h"

#include "fits/fitsloadersettings.h"

#include "safecast.h"

#include <fitsio.h>

#include <stdexcept>

namespace fitshelper
{
QMutex globalMutex;

QMutex *fitshelper::FitsHelper::getMutex()
{
	return &globalMutex;
}

int valueTypeToFitsType(ValueType valueType)
{
	switch (valueType) {
	case ValueTypeChar:
		return TSBYTE;
	case ValueTypeShort:
		return TSHORT;
	case ValueTypeLong:
		return TLONG;
	case ValueTypeLongLong:
		return TLONGLONG;
	case ValueTypeUChar:
		return TBYTE;
	case ValueTypeUShort:
		return TUSHORT;
	case ValueTypeULong:
		return TULONG;
	case ValueTypeFloat:
		return TFLOAT;
	case ValueTypeDouble:
		return TDOUBLE;
	default:
		throw std::range_error("Invalid value type, while reading fits file.");
	}
}

fitsfile *toFits(void *ptr)
{
	return static_cast<fitsfile *>(ptr);
}

FitsHelper::FitsHelper(const QString &filename)
	: lock(getMutex())
	, fitsfile(0)
{
	int status = 0;
	::fitsfile *fits = 0;
	fits_open_file(&fits, filename.toLocal8Bit().constData(), READONLY, &status);
	if (status)
		throw std::runtime_error("Could not open fits file..");
	this->fitsfile = fits;
	getMutex()->unlock();
}

FitsHelper::~FitsHelper()
{
	getMutex()->lock();
	int status = 0;
	fits_close_file(toFits(fitsfile), &status);
}

size_t FitsHelper::_getDimensions() const
{
	int status = 0;
	int dimensions;
	fits_get_img_dim(toFits(fitsfile), &dimensions, &status);
	if (status)
		throw std::runtime_error("Could not read fits image dimensions.");
	return dimensions;
}

void FitsHelper::_getSizes(size_t *sizes, size_t dimensions) const
{
	int status = 0;
	int fitsDimensions = 0;
	fits_get_img_dim(toFits(fitsfile), &fitsDimensions, &status);
	if (status)
		throw std::runtime_error("Could not read fits image size.");
	std::vector<LONGLONG> fitsSizes(fitsDimensions);
	fits_get_img_sizell(toFits(fitsfile), fitsDimensions, fitsSizes.data(), &status);
	for (size_t i = 0; i < std::min(size_t(fitsDimensions), dimensions); ++i)
		sizes[i] = fitsSizes[i];
	for (size_t i = fitsDimensions; i < dimensions; ++i)
		sizes[i] = 1;
}

void FitsHelper::_readContiguous(ValueType valueType, void *data, size_t count) const
{
	int fitsValueType = valueTypeToFitsType(valueType);
	int status = 0;
	fits_read_img(toFits(fitsfile), fitsValueType, 1, count, 0, data, 0, &status);
	if (status)
		throw std::runtime_error("Could not read fits image."); // TODO: Replace by better exception
}

void FitsHelper::_readContiguousSubimage(ValueType valueType, void *data, size_t *start, size_t *sizes, size_t dimensions) const
{
	int fitsValueType = valueTypeToFitsType(valueType);
	int status = 0;
	int fitsDimensions = _getDimensions();
	if (dimensions > size_t(fitsDimensions))
		throw std::runtime_error("Too few dimensions in fits image."); // TODO: Replace by better exception

	std::vector<long> first(fitsDimensions, 1);
	std::vector<long> last(fitsDimensions, 1);
	for (size_t i = 0; i < dimensions; ++i) {
		first[i] = start[i] + 1;
		last[i] = start[i] + sizes[i];
	}
	std::vector<long> increase(fitsDimensions, 1);
	fits_read_subset(toFits(fitsfile), fitsValueType, first.data(), last.data(), increase.data(), 0, data, 0, &status);
	if (status)
		throw std::runtime_error("Could not read fits subimage."); // TODO: Replace by better exception
}

QString FitsHelper::_readKey(const QString &name, QString *outComment) const
{
	int status = 0;
	std::array<char, FLEN_VALUE> value = {0};
	std::array<char, FLEN_COMMENT> comment = {0};
	fits_read_keyword(toFits(fitsfile), name.toLatin1().constData(), value.data(), comment.data(), &status);
	if (outComment)
		*outComment = QString::fromLatin1(comment.data());
	if (status == VALUE_UNDEFINED)
		return QString();
	if (status)
		throw std::runtime_error("Could not read fits keyword.");
	return QString::fromLatin1(value.data());
}

void FitsHelper::_readKey(int index, QString &name, QString &outValue, QString *outComment, bool *isUserKey) const
{
	int status = 0;
	std::array<char, FLEN_CARD> headerRecord;
	fits_read_record(toFits(fitsfile), index, headerRecord.data(), &status);

	std::array<char, FLEN_KEYWORD> keyword = {0};
	std::array<char, FLEN_VALUE> value = {0};
	std::array<char, FLEN_COMMENT> comment = {0};
	int keywordLength = 0;
	fits_get_keyname(headerRecord.data(), keyword.data(), &keywordLength, &status);
	fits_parse_value(headerRecord.data(), value.data(), comment.data(), &status);
	name = QString::fromLatin1(keyword.data(), keywordLength);
	outValue = QString::fromLatin1(value.data());
	if (outComment)
		*outComment = QString::fromLatin1(comment.data());
	if (isUserKey)
		*isUserKey = (fits_get_keyclass(headerRecord.data()) == TYP_USER_KEY);

	if (status)
		throw std::runtime_error("Could not read fits keyword.");
}

QMap<QString, QString> FitsHelper::_readUserKeys(QMap<QString, QString> *comments) const
{
	int status = 0;
	int keyCount = 0;
	fits_get_hdrspace(toFits(fitsfile), &keyCount, 0, &status);
	if (status)
		throw std::runtime_error("Could not get fits keyword count.");

	QMap<QString, QString> map;
	for (int i = 0; i < keyCount; ++i) {
		QString name;
		QString value;
		QString comment;
		bool isUserKey;
		_readKey(i, name, value, &comment, &isUserKey);
		if (!isUserKey)
			continue;
		map.insert(name, value);
		if (comments)
			comments->insert(name, comment);
	}
	return map;
}

TomoFitsHelper::TomoFitsHelper(const QString &filename)
	: FitsHelper(filename)
	, filename(filename)
{
}

double TomoFitsHelper::getAngle(const FitsLoaderSettings &settings) const
{
	int status = 0;
	int keyCount = 0;
	fits_get_hdrspace(toFits(fitsfile), &keyCount, 0, &status);
	if (status)
		throw std::runtime_error("Could not get fits keyword count.");

	for (const QString &key : settings.angle_fits_keys) {
		try {
			QString value = _readKey(key);
			bool ok;
			double angle = value.toDouble(&ok);
			if (ok)
				return angle;
		} catch (...) {
		}
	}
	for (const QString &pattern : settings.angle_file_patterns) {
		QRegExp rx(pattern);
		if (rx.indexIn(filename) >= 0) {
			bool ok;
			double angle = rx.cap(rx.captureCount() > 0 ? 1 : 0).toDouble(&ok);
			if (ok)
				return angle;
		}
	}
	return std::nan("");
}

} // namespace fitshelper

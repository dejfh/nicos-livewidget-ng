#ifndef LW_LIVEWIDGETCHAINS_H
#define LW_LIVEWIDGETCHAINS_H

#include <memory>
#include <functional>

#include <QObject>
#include <QString>
#include <QStringList>
#include <QPixmap>
#include <QLineF>

#include "ndim/pointer.h"
#include "ndim/range.h"
#include "ndimdata/statistic.h"

namespace lw
{

class LiveWidgetChain : public QObject
{
	Q_OBJECT

public:
	enum class DataSource : int { Image = 0, DarkImage = 1, OpenBeam = 2 };

	LiveWidgetChain();
	~LiveWidgetChain();

	void setDarkImageFilenames(const QStringList &filenames);
	void setOpenBeamFilenames(const QStringList &filenames);
	void setImageFilename(const QString &filename);

	void setDataSource(DataSource source);
	void setNormalize(bool normalize);

	void setRegionOfInterest(ndim::range<2> region);
	void resetRegionOfInterest();

	void setInvert(bool invert);
	void setUseLog(bool log);
	void setUseColor(bool color);

	void setColorRange(std::pair<double, double> bounds);
	void setAutoColorRange(bool useFullRange);

	bool isProcessedDataValid() const;
	ndim::pointer<const float, 2> processedData() const;
	void setProcessedDataCallback(std::function<void(ndim::pointer<const float, 2>)> callback);

	bool isStatisticValid() const;
	const ndimdata::DataStatistic &statistic() const;
	void setStatisticCallback(std::function<void(const ndimdata::DataStatistic &)> callback);

	bool isPixmapValid() const;
	const QPixmap &pixmap() const;
	void setPixmapCallback(std::function<void(const QPixmap &)> callback);

	void setProfileLine(QLineF line);

	bool isProfileValid() const;
	ndim::pointer<const float, 1> profile() const;
	void setProfileCallback(std::function<void(ndim::pointer<const float, 1>)> callback);

	void enableProfile(bool enable = true);
	void disableProfile();

	void setZStackRegion(ndim::range<2> region);
	void setZStackFiles(const QStringList &filenames);

	bool isZStackValid() const;
	ndim::pointer<const float, 1> zStack() const;
	void setZStackCallback(std::function<void(ndim::pointer<const float, 1>)> callback);

	void enableZStack(bool enable = true);
	void disableZStack();

	bool isTestPixmapValid() const;
	const QPixmap &testPixmap() const;
	void setTestPixmapCallback(std::function<void(const QPixmap &)> callback);

	void start();

	void getStatus(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description) const;
	QStringList getDescriptions() const;

signals:
	void validationStarted();
	void validationStep();
	void validationDone();
	void invalidated();

private slots:
	void validator_validationStarted();
	void validator_validationStep();
	void validator_validationCompleted();
	void validator_invalidated();

private:
	struct LiveWidgetChainPrivate;
	std::unique_ptr<LiveWidgetChainPrivate> p;
};

} // namespace lw

#endif // LW_LIVEWIDGETCHAINS_H

#ifndef FILTER_VALIDATOR_H
#define FILTER_VALIDATOR_H

#include <cassert>
#include <atomic>
#include <memory>
#include <vector>
#include <deque>

#include <QFuture>
#include <QFutureWatcher>
#include <QList>

#include <QtConcurrent/QtConcurrent>

#include "fc/datafilter.h"
#include "helper/threadsafe.h"

namespace fc
{

namespace validation
{

class Validator : public QObject, private Successor
{
	Q_OBJECT

	std::shared_ptr<int> m_dummy_ptr;

	std::vector<std::weak_ptr<const Validatable>> m_validatables; //!< List of Validatables to be validated by this \ref Validator

	std::vector<std::weak_ptr<const Validatable>> m_preparationValidatables;
	std::deque<std::shared_ptr<const Validatable>> m_queue; //!< Current queue of validatables
	QStringList m_validationSteps;							//!< Descriptions of current validation steps. Valid if m_isQueueValid is true.
	size_t m_validationDuration;							//!< Duration of the current validation queue. Valid if m_isQueueValid is true.
	ValidationProgress m_progress;							//!< Progress of the current validation.

	std::shared_ptr<const Validatable> m_activeValidatable; //!< Currently active validatable. Set by main thread.

	bool m_isWorking;				  //!< True, if there is a worker thread.
	std::atomic<bool> m_isQueueValid; //!< True, if the current queue is valid. Reset by main thread or worker thread, set by worker thread.
	bool m_rebuildQueue;			  //!< True, if the queue is valid for the active validatable, but should be rebuilded afterwards.
	bool m_enabled;					  //!< True, if the validator should automatically validate. Accessed only by main thread.
	bool m_hadException;

	QFuture<void> m_future;
	QFutureWatcher<void> m_watcher;

signals:
	void validationStarted();
	void validationStep();
	void validationCompleted();
	void invalidated();

public:
	Validator();
	virtual ~Validator();

	bool isWorking();

public slots:
	void start();
	void abort(bool wait);

public:
	void restart(bool wait);
	void add(std::shared_ptr<const Validatable> validatable);
	void remove(const Validatable *validatable);

	void state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description);
	QStringList descriptions() const;

private slots:
	void finished();
	void validate();

private:
	void prepareProc();
	void validationProc();
	void invalidate(const Predecessor *predecessor);
	void predecessorDeconstructed(const Predecessor *predecessor);

	// Successor interface
public:
	virtual void predecessorInvalidated(const Predecessor *predecessor) override;
};

} // namespace validation

} // namespace fc

#endif // FILTER_VALIDATOR_H

#ifndef FILTER_VALIDATOR_H
#define FILTER_VALIDATOR_H

#include <cassert>
#include <atomic>
#include <memory>
#include <vector>
#include <deque>
#include <thread>
#include <string>
#include <future>
#include <functional>

#include <QStringList>

#include "fc/datafilter.h"
#include "helper/threadsafe.h"
#include "helper/copyonwrite.h"

namespace fc
{

namespace validation
{

class Validator : private Successor
{
	// Accessed only by MAIN THREAD
	std::shared_ptr<int> m_dummy_ptr; //!< Dummy shared pointer. Needed to register this validator as a successor.

	std::vector<std::weak_ptr<const Validatable>> m_validatables; //!< List of Validatables to be validated by this \ref Validator

	bool m_enabled;   //!< True, if the validator should automatically validate.
	bool m_isWorking; //!< True, if there is a worker thread.
	bool m_rebuildQueue; //!< True, if the preparation should rerun after the current validation returns.

	// Accessed by MAIN THREAD and WORKER THREAD
	std::vector<std::weak_ptr<const Validatable>>
		m_preparationValidatables; //!< Copy of \ref m_validatables to be used by the worker thread. Written by main thread, used during preparation.
	std::shared_ptr<const Validatable> m_activeValidatable; //!< Currently active validatable. Written by main thread, used during validation.

	std::atomic<bool>
		m_isQueueValid; //!< True, if the current queue is valid (After preparation, before invalidation). Reset by main thread, set by worker thread.
	std::deque<std::shared_ptr<const Validatable>> m_queue; //!< Current queue of validatables. Written by worker thread. Valid if \ref m_isQueueValid is true.
	QStringList m_validationSteps; //!< Descriptions of current validation steps. Written by worker thread. Valid if \ref m_isQueueValid is true.
	size_t m_validationDuration;   //!< Duration of the current validation queue. Written by worker thread. Valid if \ref m_isQueueValid is true.

	ValidationProgress m_progress; //!< Progress of the current validation. Reset by main thread, written by worker thread.

	bool m_hadException; //!< True, if the last run of the worker thread exited with an exception. Written by worker thread.

	std::future<void> m_future;

protected:
	virtual void invokeFinished() = 0;
	virtual void onValidationStarted() = 0;
	virtual void onValidationStep() = 0;
	virtual void onValidationComplete() = 0;
	virtual void onInvalidated() = 0;

public:
	Validator();
	virtual ~Validator();

	bool isWorking() const;

	void start();
	void abort(bool wait);

	void restart(bool wait);
	void add(std::shared_ptr<const Validatable> validatable);
	void remove(const Validatable *validatable);

	void state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description);
	QStringList descriptions() const;

protected:
	void finished();

private:
	void validate();

private:
	void prepareProc();
	void validationProc();
	void invalidate(const Predecessor *predecessor);
	void predecessorDeconstructed(const Predecessor *predecessor);

	// Successor interface
private:
	virtual void predecessorInvalidated(const Predecessor *predecessor) override;
};

} // namespace validation

} // namespace fc

#endif // FILTER_VALIDATOR_H

#include "fc/validation/validator.h"

#include <iostream>

#include <QStringList>

namespace fc
{

namespace validation
{

class ValidatorPreparationProgress : public PreparationProgress
{
public:
	std::deque<std::shared_ptr<const Validatable>> queue;
	QStringList steps;
	size_t duration;

	ValidatorPreparationProgress()
		: duration(0)
	{
	}

	virtual void appendValidatable(std::shared_ptr<const Validatable> filter) override
	{
		queue.push_back(std::move(filter));
	}
	virtual bool containsValidatable(const Validatable *filter) const override
	{
		for (const std::shared_ptr<const Validatable> &item : queue)
			if (item.get() == filter)
				return true;
		return false;
	}

	virtual void addStep(size_t flops, const QString &description) override
	{
		steps.append(description);
		duration += flops;
	}
};

Validator::Validator()
	: m_enabled(false)
	, m_isWorking(false)
	, m_rebuildQueue(false)
	, m_activeValidatable(nullptr)
	, m_isQueueValid(false)
	, m_hadException(false)
{
	m_dummy_ptr = std::make_shared<int>();
}

Validator::~Validator()
{
	m_enabled = false;
	m_progress.cancel();
	if (m_isWorking)
		m_future.wait();
	m_isWorking = false;
	for (const std::weak_ptr<const Validatable> &weak : m_validatables) {
		std::shared_ptr<const Validatable> shared = weak.lock();
		if (shared)
			shared->removeSuccessor(this);
	}
}

bool Validator::isWorking() const
{
	return m_isWorking;
}

void Validator::start()
{
	m_enabled = true;
	this->invokeFinished();
}

void Validator::abort(bool wait)
{
	m_enabled = false;
	m_progress.cancel();
	m_rebuildQueue = true;
	m_isQueueValid = false;
	if (wait && m_isWorking) {
		m_future.wait();
		this->finished();
	}
}

void Validator::restart(bool wait)
{
	bool wasEnabled = m_enabled;
	abort(wait);
	if (wasEnabled)
		start();
}

void Validator::add(std::shared_ptr<const Validatable> validatable)
{
	m_validatables.emplace_back(validatable);
	m_rebuildQueue = true;
	std::shared_ptr<Successor> successor(m_dummy_ptr, this);
	validatable->addSuccessor(successor);
	onInvalidated();
}

void Validator::remove(const Validatable *validatable)
{
	validatable->removeSuccessor(this);
	auto emptyOrSame = [validatable](const std::weak_ptr<const Validatable> &ptr) { return ptr.expired() || ptr.lock().get() == validatable; };
	m_validatables.erase(std::remove_if(m_validatables.begin(), m_validatables.end(), emptyOrSame), m_validatables.end());
	m_rebuildQueue = true;
}

void Validator::state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description)
{
	progress = 0;
	duration = 0;
	step = 0;
	stepCount = 0;
	if (!m_isWorking && !m_hadException) {
		description = QString("done.");
		return;
	}
	if (!m_isQueueValid) {
		stepCount = m_preparationValidatables.size();
		step = m_progress.currentStep();
		description = QString("preparing...");
	} else {
		progress = m_progress.progress();
		duration = m_validationDuration;
		step = m_progress.currentStep();
		stepCount = m_validationSteps.size();
		if (step < stepCount)
			description = m_validationSteps[step];
		else
			description = QString("finishing...");
	}
	if (m_hadException)
		description = QString("ERROR while ").append(description);
}

QStringList Validator::descriptions() const
{
	if (!m_isQueueValid.load(std::memory_order_acquire))
		return QStringList();
	return m_validationSteps;
}

void Validator::validate()
{
	if (!m_enabled || m_isWorking || m_hadException)
		return;
	if (m_rebuildQueue || !m_isQueueValid.load(std::memory_order_acquire)) {
		onValidationStarted();
		m_rebuildQueue = false;
		m_activeValidatable = 0;
		m_progress.reset();
		m_isWorking = true;
		m_preparationValidatables.clear();
		std::copy(m_validatables.cbegin(), m_validatables.cend(), std::back_inserter(m_preparationValidatables));
		m_future = std::async(std::launch::async, [this]() { this->prepareProc(); });
		return;
	}
	if (!m_queue.empty()) {
		m_activeValidatable = std::move(m_queue.front());
		m_queue.pop_front();
		m_isWorking = true;
		m_future = std::async(std::launch::async, [this]() { this->validationProc(); });
		return;
	}
	onValidationComplete();
}
void Validator::prepareProc()
{
	ValidatorPreparationProgress progress;
	try {
		for (const std::weak_ptr<const Validatable> &weak : m_preparationValidatables) {
			auto validatable = weak.lock();
			if (validatable)
				validatable->prepareValidation(progress);
			m_progress.advanceStep();
			m_progress.throwIfCancelled();
		}
		m_queue = std::move(progress.queue);
		m_validationSteps = progress.steps;
		m_validationDuration = progress.duration;
		m_isQueueValid.store(true, std::memory_order_release);
	} catch (const hlp::OperationCanceledException &) {
		return;
	} catch (const std::exception &ex) {
		m_hadException = true; // TODO: What happens on an exception?
		std::cerr << "Validation exception: " << ex.what() << std::endl;
	} catch (...) {
		std::cerr << "Unknown Validation exception." << std::endl;
		m_hadException = true;
	}
	m_progress.reset();
	this->invokeFinished();
}

void Validator::validationProc()
{
	try {
		m_activeValidatable->validate(m_progress);
		assert(m_activeValidatable->isValid()); // TODO: Replace by exception?
	} catch (const hlp::OperationCanceledException &) {
		return;
	} catch (const std::exception &ex) {
		m_hadException = true; // TODO: What happens on an exception?
		std::cerr << "Validation exception: " << ex.what() << std::endl;
	} catch (...) {
		std::cerr << "Unknown Validation exception." << std::endl;
		m_hadException = true;
	}
	this->invokeFinished();
}

void Validator::finished()
{
	if (m_isWorking) {
		if (m_future.wait_for(std::chrono::seconds::zero()) != std::future_status::ready)
			return;
		else {
			m_future.get();
			m_activeValidatable = nullptr;
			m_isWorking = false;
			onValidationStep();
		}
	}
	validate();
}

void Validator::predecessorInvalidated(const Predecessor *predecessor)
{
	if (m_isWorking && (!m_activeValidatable || predecessor == m_activeValidatable.get()))
		restart(false);
	else
		m_rebuildQueue = true;
	m_hadException = false;
	this->invokeFinished();
	onInvalidated();
}

} // namespace validation

} // namespace fc

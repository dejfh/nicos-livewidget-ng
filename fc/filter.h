#ifndef FILTER_FILTER_H
#define FILTER_FILTER_H

#include <memory> // std::shared_ptr, std::weak_ptr

#include <QString>

#include "helper/asyncprogress.h"

#include "variadic.h"

/*!
 * \ingroup Filter
 * @{
 */

namespace fc
{

class Predecessor;
class Successor;
class Validatable;
class PreparationProgress;
class ValidationProgress;

/*! \brief An object depending on the output of a filter in a filterchain, e.g. successing filters.
 *
 * Gets notified by the given predecessor, when it is invalidated or deconstructed.
 *
 * \remarks \ref filter::validation::Validator implements Successor to trigger revalidation.
 * */
class Successor
{
public:
	/*! \brief Notifies about a predecessor being invalidated.
	 *
	 * If the successor currently depends on the predecessor it must invalidate itself and its successors.
	 *
	 * #### Threading
	 * This method is called only from the \b main thread.
	 * */
	virtual void predecessorInvalidated(const Predecessor *predecessor) = 0;
};

/*! \brief A predecessor in a filterchain
 *
 * Notifies successors, when it is invalidated or deconstructed.
 * A filter is invalidated, whenever a call to \ref DataFilter::getData or \ref DataFilter::getConstData may give other results than before.
 * */
class Predecessor
{
public:
	/*! \brief Adds a successor.
	 *
	 * #### Threading
	 * This method is called only from the \b main thread.
	 * */
	virtual void addSuccessor(std::weak_ptr<Successor> successor) const = 0;
	/*! \brief Removes a successor.
	 *
	 * #### Threading
	 * This method is called only from the \b main thread.
	 * */
	virtual void removeSuccessor(Successor *successor) const = 0;
};

/*! \brief A filter that can be validated.
 *
 * This interface is used by Validator to initiate the preparation and validation processes.
 * */
class Validatable : public virtual Predecessor
{
public:
	/*! \brief Calculates the estamited duration of a validation.
	 *
	 * \param progress Notifies about cancelation.
	 * \param counter Accumulates the duration.
	 *
	 * A call to this method initiates the preparation process.
	 *
	 * \remarks This method is also available in DataFilter.
	 *
	 * #### Threading
	 * This method is called only from the \b worker thread.
	 * */
	virtual void prepareValidation(PreparationProgress &progress) const = 0;

	/*! \brief Validates filter.
	 * \param progress Notifies about cancelation and tracks progress.
	 *
	 * A call to this method initiates the validation process.
	 *
	 * #### Threading
	 * This method is called only from the \b worker thread.
	 * */
	virtual void validate(ValidationProgress &progress) const = 0;

	/*! \brief Checks if the filter is currently valid.
	 *
	 * #### Threading
	 * This method may be called by \b any thread.
	 * */
	virtual bool isValid() const = 0;
};

class PreparationProgress : public hlp::AsyncProgress<size_t>
{
public:
	virtual void appendValidatable(std::shared_ptr<const Validatable> fc) = 0;

	virtual bool containsValidatable(const Validatable *fc) const = 0;

	virtual void addStep(size_t flops, const QString &description) = 0;
};

/*! \brief Notifies about cancelation and collects the progress during the validation process.
 *
 * #### Threading
 * All methods are threadsafe and may be called by \b any thread.
 * */
class ValidationProgress : public hlp::AsyncProgress<size_t>
{
protected:
	std::atomic<size_t> m_step;

	std::vector<std::shared_ptr<const void>> m_forcedRefs;

public:
	ValidationProgress()
		: m_step(0)
	{
	}
	ValidationProgress(const ValidationProgress &other)
		: AsyncProgress(other)
		, m_step(other.m_step.load())
	{
	}
	ValidationProgress &operator=(const ValidationProgress &other)
	{
		static_cast<AsyncProgress &>(*this) = other;
		m_step = other.m_step.load();
		return *this;
	}

	void reset()
	{
		AsyncProgress::reset();
		m_step = 0;
		m_forcedRefs.clear();
	}
	void cancel()
	{
		AsyncProgress::cancel();
	}

	void advanceStep()
	{
		++m_step;
	}
	size_t currentStep()
	{
		return m_step;
	}

	void holdRef(std::shared_ptr<const void> ptr)
	{
		m_forcedRefs.push_back(std::move(ptr));
	}
};

} // namespace fc

//! @}

#endif // FILTER_FILTER_H

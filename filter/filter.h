#ifndef DATA_FILTER_H
#define DATA_FILTER_H

#include <exception>
#include <memory> // std::shared_ptr, std::weak_ptr

#include <QString>

#include "asyncprogress.h"

#include "variadic.h"
#include "safecast.h"

#include "filter/ndimcontainer.h"

/*!
 * \ingroup Filter
 * @{
 */

namespace filter
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
	virtual void prepare(PreparationProgress &progress) const = 0;

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

class PreparationProgress : public AsyncProgress
{
public:
	virtual void appendValidatable(std::shared_ptr<const Validatable> filter) = 0;

	virtual bool containsValidatable(const Validatable *filter) const = 0;

	virtual void addStep(size_t flops, const QString &description) = 0;
};

/*! \brief Notifies about cancelation and collects the progress during the validation process.
 *
 * #### Threading
 * All methods are threadsafe and may be called by \b any thread.
 * */
class ValidationProgress : public AsyncProgress
{
protected:
	std::atomic<size_t> m_step;

public:
	ValidationProgress()
		: m_step(0)
	{
	}
	void reset()
	{
		AsyncProgress::reset();
		m_step = 0;
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
};

///*! \brief Collects the time consuming steps and their estimated duration.
// *
// * Is used during the preparation process.
// * */
// struct DurationCounter {
//	/*! \brief Apands the current filter as a Validatable to the Validationchain.
//	 *
//	 * \param filter The filter to be appended.
//	 *
//	 * #### Threading
//	 * This method is called only from the \b worker thread during the \b preparation process.
//	 * */
//	virtual void appendValidatable(std::shared_ptr<const Validatable> filter) = 0;

//	/*! \brief Checks if the current filter already is in the Validationchain.
//	 *
//	 * \param filter The filter to search for.
//	 *
//	 * This is used by buffers to check if they would already be validated during the validation process.
//	 *
//	 * #### Threading
//	 * This method is called only from the \b worker thread during the \b preparation process.
//	 * */
//	virtual bool containsValidatable(const Validatable *filter) const = 0;

//	/*! \brief Adds the current filter as a time consuming step.
//	 *
//	 * \param flops The estimated duration in flops.
//	 *
//	 * \param description A short description of the time consuming step.
//	 *
//	 * \c flops must match the exact progress that will be added during the validation process.
//	 *
//	 * #### Threading
//	 * This method is called only from the \b worker thread during the \b preparation process.
//	 * */
//	virtual void addStep(size_t flops, const QString &description) = 0;
//};

template <typename _FilterType>
struct OverloadDummy {
};

template <typename _ElementType, size_t _Dimensionality = 0>
class DataFilter : public virtual Predecessor
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

	/*!
	 * \brief prepare
	 * \param progress Collects validation steps and notifies about cancelation.
	 * \return Returns the sizes in each dimension of the returned data.
	 */
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress,
		OverloadDummy<DataFilter<ElementType, Dimensionality>> = OverloadDummy<DataFilter<ElementType, Dimensionality>>()) const = 0;
	/*!
	 * \brief getData
	 * \param progress Collects progress information and notifies about cancelation.
	 * \param result The container to receive the data. If the container size does not match the needed size, the container will be resized.
	 */
	virtual void getData(ValidationProgress &progress, Container<ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>> = OverloadDummy<DataFilter<ElementType, Dimensionality>>()) const = 0;

	virtual ndim::sizes<Dimensionality> prepareConst(PreparationProgress &progress,
		OverloadDummy<DataFilter<ElementType, Dimensionality>> = OverloadDummy<DataFilter<ElementType, Dimensionality>>()) const = 0;
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>> = OverloadDummy<DataFilter<ElementType, Dimensionality>>()) const = 0;
};

template <typename _ElementType, size_t _Dimensionality>
DataFilter<_ElementType, _Dimensionality> &asDataFilter(DataFilter<_ElementType, _Dimensionality> &filter)
{
	return filter;
}
template <typename _ElementType, size_t _Dimensionality>
const DataFilter<_ElementType, _Dimensionality> &asDataFilter(const DataFilter<_ElementType, _Dimensionality> &filter)
{
	return filter;
}

template <typename _FilterType>
struct AsDataFilter {
	using type = typename std::remove_reference<decltype(asDataFilter(std::declval<_FilterType>()))>::type;
};
template <typename _FilterType>
using AsDataFilter_t = typename AsDataFilter<_FilterType>::type;

template <typename _FilterType>
std::shared_ptr<AsDataFilter_t<_FilterType>> asDataFilter(std::shared_ptr<_FilterType> filter)
{
	return filter;
}

} // namespace filter

//! @}

#endif // DATA_FILTER_H

#ifndef ASYNCPROGRESS_H
#define ASYNCPROGRESS_H

#include <cstddef>
#include <atomic>
#include <stdexcept>

class OperationCanceledException : public std::runtime_error
{
public:
	OperationCanceledException()
		: std::runtime_error("Validation cancelled.")
	{
	}
};

class AsyncTaskControl
{
	std::atomic<bool> m_canceled;

public:
	AsyncTaskControl()
		: m_canceled(false)
	{
	}

	bool isCanceled() const
	{
		return m_canceled;
	}

	void cancel()
	{
		m_canceled = true;
	}

	void throwIfCanceled() const
	{
		if (m_canceled)
			throw OperationCanceledException();
	}
};

template <typename _ProgressType>
class AsyncProgressControl : public AsyncTaskControl
{
public:
	using ProgressType = _ProgressType;

private:
	std::atomic<_ProgressType> m_progress;

public:
	AsyncProgressControl(ProgressType progress = ProgressType())
		: m_progress(progress)
	{
	}

	ProgressType progress() const
	{
		return m_progress;
	}

	void advanceProgress(ProgressType progress)
	{
		m_progress += progress;
	}
};

/*! \brief Notifies about cancelation and collects the progress during an async process.
 * */
class AsyncProgress
{
	std::atomic<size_t> m_progress;
	std::atomic<bool> m_cancelled;

public:
	AsyncProgress()
		: m_progress(0)
		, m_cancelled(false)
	{
	}

	void advanceProgress(size_t flops)
	{
		m_progress.fetch_add(flops, std::memory_order_relaxed);
	}

	bool isCancelled() const
	{
		return m_cancelled;
	}
	size_t progress() const
	{
		return m_progress;
	}

	void throwIfCancelled() const
	{
		if (this->isCancelled())
			throw OperationCanceledException();
	}

	/*! \brief Resets the progress to uncanceled and zero.
	 * */
	void reset()
	{
		m_cancelled = false;
		m_progress = 0;
	}

	/*! \brief Sets the progress to the canceled state.
	 * */
	void cancel()
	{
		m_cancelled = true;
	}
};

class AsyncProgressMaster
{
public:
	virtual bool canceled() const = 0;
	virtual bool setProgress(float progress) = 0;
};

class AsyncFloatProgress
{
	AsyncProgressMaster *m_parent;
	float m_progress;
	float m_step;

public:
	AsyncFloatProgress()
		: m_parent(0)
		, m_progress(0)
		, m_step(1)
	{
	}
	AsyncFloatProgress(AsyncProgressMaster *parent, float start = 0.0, float step = 1.0)
		: m_parent(parent)
		, m_progress(start)
		, m_step(step)
	{
	}

	bool step(float amount)
	{
		if (!m_parent)
			return false;
		m_progress += amount * m_step;
		return m_parent->setProgress(m_progress);
	}

	bool canceled() const
	{
		if (!m_parent)
			return false;
		return m_parent->canceled();
	}

	AsyncFloatProgress sub(float amount)
	{
		float progress = m_progress;
		m_progress += amount * m_step;
		return AsyncFloatProgress(m_parent, progress, amount * m_step);
	}
};

#endif // ASYNCPROGRESS_H

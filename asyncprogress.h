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

// class AsyncTaskControl
//{
//	std::atomic<bool> m_canceled;

// public:
//	AsyncTaskControl()
//		: m_canceled(false)
//	{
//	}

//	bool isCanceled() const
//	{
//		return m_canceled;
//	}

//	void cancel()
//	{
//		m_canceled = true;
//	}

//	void throwIfCanceled() const
//	{
//		if (m_canceled)
//			throw OperationCanceledException();
//	}
//};

// template <typename _ProgressType>
// class AsyncProgressControl : public AsyncTaskControl
//{
// public:
//	using ProgressType = _ProgressType;

// private:
//	std::atomic<_ProgressType> m_progress;

// public:
//	AsyncProgressControl(ProgressType progress = ProgressType())
//		: m_progress(progress)
//	{
//	}

//	ProgressType progress() const
//	{
//		return m_progress;
//	}

//	void advanceProgress(ProgressType progress)
//	{
//		m_progress += progress;
//	}
//};

class AsyncTask
{
protected:
    std::atomic<bool> m_cancelled;

public:
    AsyncTask()
        : m_cancelled(false)
    {
    }
    AsyncTask(const AsyncTask &other)
        : m_cancelled(other.m_cancelled.load())
    {
    }
    AsyncTask &operator=(const AsyncTask &other)
    {
        this->m_cancelled = other.m_cancelled.load();
        return *this;
    }

    bool isCancelled() const
    {
        return m_cancelled;
    }

    void throwIfCancelled() const
    {
        if (this->isCancelled())
            throw OperationCanceledException();
    }

    void reset()
    {
        m_cancelled = false;
    }
    void cancel()
    {
        m_cancelled = true;
    }
};

/*! \brief Notifies about cancelation and collects the progress during an async process.
 * */
template <typename _ValueType>
class AsyncProgress : public AsyncTask
{
public:
    using ValueType = _ValueType;

protected:
    std::atomic<ValueType> m_progress;

public:
	AsyncProgress()
		: m_progress(0)
	{
	}
    AsyncProgress(const AsyncProgress &other)
        : AsyncTask(other)
        , m_progress(other.m_progress.load())
    {
    }
    AsyncProgress &operator=(const AsyncProgress &other)
    {
        static_cast<AsyncTask &>(*this) = other;
        m_progress = other.m_progress.load();
        return *this;
    }

    void reset()
    {
        AsyncTask::reset();
        m_progress = 0;
    }
    void cancel()
    {
        AsyncTask::cancel();
    }

    void advanceProgress(ValueType flops)
	{
        flops += m_progress.load(std::memory_order_relaxed);
        m_progress.store(flops, std::memory_order_relaxed);
	}
    void setProgress(ValueType progress)
    {
        m_progress.store(progress, std::memory_order_relaxed);
    }

    ValueType progress() const
	{
		return m_progress;
	}
};

class AsyncPartialProgress
{
protected:
    AsyncProgress<float> *progress;
    float factor;
    float final;

public:
    AsyncPartialProgress(AsyncProgress<float> *progress, float step)
        : progress(progress)
        , factor(step)
        , final(progress ? progress->progress() + step : step)
    {
    }
    ~AsyncPartialProgress()
    {
        if (progress)
            progress->setProgress(final);
    }
    AsyncPartialProgress(const AsyncPartialProgress &other) = delete;

    AsyncPartialProgress(AsyncPartialProgress &&other)
        : progress(other.progress)
        , factor(other.factor)
        , final(other.final)
    {
        other.progress = nullptr;
    }
    AsyncPartialProgress &operator=(AsyncPartialProgress &&other)
    {
        if (progress)
            progress->setProgress(final);
        progress = other.progress;
        factor = other.factor;
        final = other.final;
        other.progress = nullptr;
        return *this;
    }

    bool isCancelled() const
    {
        return progress ? progress->isCancelled() : false;
    }

    void throwIfCancelled() const
    {
        if (progress)
            progress->throwIfCancelled();
    }

    void advance(float step)
    {
        if (progress)
            progress->advanceProgress(step * factor);
    }

    AsyncPartialProgress sub(float step)
    {
        return AsyncPartialProgress(progress, step);
    }
};

// class AsyncProgressMaster
//{
// public:
//	virtual bool canceled() const = 0;
//	virtual bool setProgress(float progress) = 0;
//};

// class AsyncFloatProgress
//{
//	AsyncProgressMaster *m_parent;
//	float m_progress;
//	float m_step;

// public:
//	AsyncFloatProgress()
//		: m_parent(0)
//		, m_progress(0)
//		, m_step(1)
//	{
//	}
//	AsyncFloatProgress(AsyncProgressMaster *parent, float start = 0.0, float step = 1.0)
//		: m_parent(parent)
//		, m_progress(start)
//		, m_step(step)
//	{
//	}

//	bool step(float amount)
//	{
//		if (!m_parent)
//			return false;
//		m_progress += amount * m_step;
//		return m_parent->setProgress(m_progress);
//	}

//	bool canceled() const
//	{
//		if (!m_parent)
//			return false;
//		return m_parent->canceled();
//	}

//	AsyncFloatProgress sub(float amount)
//	{
//		float progress = m_progress;
//		m_progress += amount * m_step;
//		return AsyncFloatProgress(m_parent, progress, amount * m_step);
//	}
//};

#endif // ASYNCPROGRESS_H

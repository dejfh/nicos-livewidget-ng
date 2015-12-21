//#include "signalingprogress.h"

//SignalingProgress::SignalingProgress(QObject *parent, int factor)
//	: QObject(parent)
//	, m_progress(0)
//	, m_aborted(false)
//	, m_factor(factor)
//{
//}

//void SignalingProgress::abort()
//{
//	m_aborted = true;
//}

//double SignalingProgress::progress() const
//{
//	return m_progress;
//}

//bool SignalingProgress::canceled() const
//{
//	return m_aborted;
//}

//bool SignalingProgress::setProgress(float progress)
//{
//	m_progress = progress;
//	emit(progressChanged(int(progress * m_factor)));
//	return m_aborted;
//}

//AsyncFloatProgress SignalingProgress::subProgress(float amount)
//{
//	return AsyncFloatProgress(this, m_progress, amount);
//}

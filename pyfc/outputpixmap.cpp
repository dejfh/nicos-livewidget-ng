#include "outputpixmap.h"

namespace pyfc
{

OutputPixmap::OutputPixmap(fc::validation::QtValidator *validator)
	: OutputFilterBase(validator)
{
	m_buffer = std::make_shared<fc::filter::Buffer<QPixmap>>();
	validator->add(m_buffer);
}

OutputPixmap::~OutputPixmap()
{
	this->validator()->remove(m_buffer.get());
}

const fc::Validatable &OutputPixmap::ownValidatable() const
{
	return *m_buffer;
}

void OutputPixmap::emitValidated()
{
	emit validated(m_buffer->data().first());
}

} // namespace pyfc

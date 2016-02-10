#ifndef PYFC_OUTPUTPIXMAP_H
#define PYFC_OUTPUTPIXMAP_H

#include <memory>

#include <QObject>
#include <QPixmap>

#include "fc/filter/buffer.h"

#include "pyfc/outputfilter.h"

namespace pyfc
{

class OutputPixmap : public OutputFilterBase
{
	Q_OBJECT

	std::shared_ptr<fc::filter::Buffer<QPixmap>> m_buffer;

public:
	explicit OutputPixmap(fc::validation::QtValidator *validator);
	~OutputPixmap();

signals:
	void validated(QPixmap pixmap);

protected:
	virtual const fc::Validatable &ownValidatable() const override;
	virtual void emitValidated() override;
};

} // namespace pyfc

#endif // PYFC_OUTPUTPIXMAP_H

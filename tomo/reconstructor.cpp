#include "tomo/reconstructor.h"

#include "tomo/shaders.h"

#include <vector>
#include <array>
#include <string>
#include <algorithm>

#include <QGLFunctions>
#include <QGLContext>
#include <QThread>

#include <QMatrix4x4>
#include <QVector2D>
#include <QTransform>

#include <QGLBuffer>

#include "helper/helper.h"

#include "ndim/algorithm.h"

const float PI = 3.1415926535897932384626433832795f;

namespace tomo
{

size_t upper_power_of_two(size_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;
	return v;
}

using GL::Program;
using GL::ProgramGroup;

class Reconstructor::ReconstructorPrivate
{
public:
	QGLFunctions gl;

	GL::ProgramGroup m_programs;

	unsigned int m_filledSinogram;
	unsigned int m_capacitySinogram;
	unsigned int m_resolutionSinogram;
	unsigned int m_resolutionReconstruction;

	// angles in degrees
	float m_sinogramCenter;

	float m_stepSize;
	float m_likelihood;
	float m_likelihoodTest;

    ReconstructorPrivate(const QGLContext &context)
        : gl(&context)
		, m_programs(gl)
	{
	}

	void guess_toRecon_fromSinoOrg()
	{
		/*
		 * Backproject every projection into the volume
		 * */
		Program &program = m_programs.use(GL::ProgGuess_ToRecon_FromSinoOrg);
		gl.glUniform1f(program.loc_var[GL::Var_Center], m_sinogramCenter / m_resolutionSinogram);
		gl.glUniform1f(program.loc_var[GL::Var_Step], float(m_filledSinogram) * float(m_resolutionReconstruction));
		GLint count = m_filledSinogram;
		for (GLint i = 0; i < count; ++i) {
			gl.glUniform1f(program.loc_var[GL::Var_Projection], (i + .5f) / m_capacitySinogram);
			m_programs.draw();
		}
	}

	void mask_toReconTest_fromRecon()
	{
		/*
		  * Mask the volume
		  * */
		Program &program = m_programs.use(GL::ProgMask_ToReconTest_FromRecon);
		hlp::unused(program);
		m_programs.draw();
	}

	void sum_toReconTest_fromReconAndGradient(float stepSize)
	{
		/*
		 * reconstruction test = mask ( reconstruction + step * gradient )
		 * */
		Program program = m_programs.use(GL::ProgSum_ToReconTest_FromReconAndGradient);
		gl.glUniform1f(program.loc_var[GL::Var_Step], stepSize);
		m_programs.draw();
		GL::assert_glError();
	}

	void project_toSinoTest_fromReconTest()
	{
		Program program = m_programs.use(GL::ProgProject_ToSinoTest_FromReconTest);
		gl.glUniform2f(program.loc_var[GL::Var_SinoRange], 1.f, float(m_filledSinogram) / float(m_capacitySinogram));
		gl.glUniform1f(program.loc_var[GL::Var_Center], m_sinogramCenter / m_resolutionSinogram);
		for (size_t i = 0; i < m_resolutionReconstruction; ++i) {
			gl.glUniform1f(program.loc_var[GL::Var_Depth], (i + .5f) / m_resolutionReconstruction);
			m_programs.draw();
		}
		GL::assert_glError();
	}

	void likelihood_fromSinoAndSinoTest()
	{
		/*
		 * Calculate the logarithm of the probability of each pixel
		 * */
		Program program = m_programs.use(GL::ProgLikelihood_FromSinoOrgAndSinoTest);
		gl.glUniform2f(program.loc_var[GL::Var_SinoRange], 1.f, float(m_filledSinogram) / float(m_capacitySinogram));
		m_programs.draw();
	}

	void sum_Likelihood()
	{
		/*
		 * Sum the logarithms of the likelihoods
		 * */
		size_t sumStep = upper_power_of_two(std::max(m_resolutionSinogram, m_filledSinogram));
		for (sumStep >>= 1; sumStep > 0; sumStep >>= 1) {
			size_t width = std::min(size_t(m_resolutionSinogram), sumStep);
			size_t height = std::min(size_t(m_filledSinogram), sumStep);

			Program program = m_programs.use(GL::ProgSum_Likelihood);
			gl.glUniform2f(program.loc_var[GL::Var_SinoRange], float(width) / float(m_resolutionSinogram), float(height) / float(m_capacitySinogram));
			gl.glUniform2f(
				program.loc_var[GL::Var_SumStep], float(sumStep) / float(m_resolutionSinogram), float(sumStep) / float(m_capacitySinogram));
			m_programs.draw();

			std::swap(m_programs.textures[GL::Tex_Likelihood], m_programs.textures[GL::Tex_Sum]);
		}
		glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, &m_likelihoodTest);
		GL::assert_glError();
	}

	void gradient_FromSinoAndSinoRecon()
	{
		/*
		 * Calculate the gradient of the quality function
		 * */
		Program program = m_programs.use(GL::ProgGradient_FromSinoOrgAndSinoRecon);
		gl.glUniform1f(program.loc_var[GL::Var_Center], m_sinogramCenter / m_resolutionSinogram);
		GLint count = m_filledSinogram;
		for (GLint i = 0; i < count; i++) {
			gl.glUniform1f(program.loc_var[GL::Var_Projection], (i + .5f) / m_capacitySinogram);
			m_programs.draw();
		}
		GL::assert_glError();
	}

	void accept_test()
	{
		std::swap(m_programs.textures[GL::Tex_Recon], m_programs.textures[GL::Tex_ReconTest]);
		std::swap(m_programs.textures[GL::Tex_SinoRecon], m_programs.textures[GL::Tex_SinoTest]);
		std::swap(m_likelihood, m_likelihoodTest);
	}
};

void Reconstructor::acceptTestDirect()
{
	m->likelihood_fromSinoAndSinoTest();
	m->sum_Likelihood();
	m->accept_test();
	m->gradient_FromSinoAndSinoRecon();
}

Reconstructor::Reconstructor(const QGLContext &context)
	: m(new ReconstructorPrivate(context))
{
	m->m_programs.createPrograms();
}

Reconstructor::~Reconstructor()
{
    clear();
}

void Reconstructor::prepare(int resolution, int sinogram_capacity, float center)
{
	m->m_resolutionSinogram = resolution;
	m->m_resolutionReconstruction = resolution;
	m->m_capacitySinogram = sinogram_capacity;

	m->m_programs.createTextures(GL::TexT_Sino, resolution, sinogram_capacity);
	m->m_programs.createTextures(GL::TexT_Volume, resolution, resolution);
	m->m_programs.createTextures(GL::TexT_SinoLine, resolution, 1);
	m->m_programs.createTextures(GL::TexT_SinoColumn, sinogram_capacity, 1);
	m->m_programs.createTextures(GL::TexT_SinoColumnMatrix, sinogram_capacity, 1);

	m->m_sinogramCenter = center;
	m->m_filledSinogram = 0;

	GL::assert_glError();
}

void Reconstructor::setOpenBeam(ndim::pointer<const quint16, 1> data)
{
	GL::assert_glError();

	assert(data.isContiguous()); // TODO: Use pixel buffers to allow other layout.
	GL::assert_glError();

	glBindTexture(GL_TEXTURE_1D, m->m_programs.textures[GL::Tex_Intensity]);
	GL::assert_glError();
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, m->m_resolutionSinogram, GL_RED, GL_UNSIGNED_SHORT, data.data);
	GL::assert_glError();
	glBindTexture(GL_TEXTURE_1D, 0);
	GL::assert_glError();
}

// angles in degrees
void Reconstructor::setSinogram(ndim::pointer<const quint16, 2> data, ndim::pointer<float, 1> angles)
{
	m->m_filledSinogram = 0;
	appendSinogram(data, angles);
}

// angles in degrees
void Reconstructor::appendSinogram(ndim::pointer<const quint16, 2> data, ndim::pointer<float, 1> angles)
{
	assert(data.height() == angles.size());
	assert(data.width() == m->m_resolutionSinogram);
	assert(m->m_filledSinogram + data.height() <= m->m_capacitySinogram);

	GL::assert_glError();
	size_t count = data.height();

	GL::assert_glError();
	{
		QGLBuffer unpackbuffer(QGLBuffer::PixelUnpackBuffer);
		GL::assert_glError();
		unpackbuffer.setUsagePattern(QGLBuffer::StaticDraw);
		unpackbuffer.create();
		unpackbuffer.bind();
		GL::assert_glError();
		unpackbuffer.allocate(int(data.size() * sizeof(quint16)));
		GL::assert_glError();
		ndim::pointer<quint16, 2> buffer_data =
			ndim::make_ptr_contiguous(static_cast<quint16 *>(unpackbuffer.map(QGLBuffer::WriteOnly)), data.width(), data.height());
		GL::assert_glError();
		ndim::copy(data, buffer_data);
		unpackbuffer.unmap();

		GL::assert_glError();

		glBindTexture(GL_TEXTURE_2D, m->m_programs.textures[GL::Tex_SinoOrg]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, m->m_filledSinogram, m->m_resolutionSinogram, GLsizei(count), GL_RED, GL_UNSIGNED_SHORT, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		unpackbuffer.release();
	}
	GL::assert_glError();
	{
		QGLBuffer unpackbuffer(QGLBuffer::PixelUnpackBuffer);
		unpackbuffer.create();
		unpackbuffer.bind();
		unpackbuffer.allocate(int(count * sizeof(float) * 2));

		float *buffer_data = hlp::cast_over_void<float *>(unpackbuffer.map(QGLBuffer::WriteOnly));
		for (size_t i = 0; i < count; ++i) {
			float angle = float(angles(i)) * PI / 180.f;
			*buffer_data = cos(angle);
			++buffer_data;
			*buffer_data = sin(angle);
			++buffer_data;
		}
		unpackbuffer.unmap();

		GL::assert_glError();

		glBindTexture(GL_TEXTURE_1D, m->m_programs.textures[GL::Tex_AngleMatrix]);
		glTexSubImage1D(GL_TEXTURE_1D, 0, m->m_filledSinogram, GLsizei(count), GL_RG, GL_FLOAT, 0);
		glBindTexture(GL_TEXTURE_1D, 0);

		unpackbuffer.release();
	}
	GL::assert_glError();

	m->m_filledSinogram += int(data.height());
	m->m_programs.sino_filled = m->m_filledSinogram;

	if (m->m_programs.textures[GL::Tex_Recon]) {
		std::swap(m->m_programs.textures[GL::Tex_Recon], m->m_programs.textures[GL::Tex_ReconTest]);
		m->project_toSinoTest_fromReconTest();

		acceptTestDirect();
	}

	GL::assert_glError();
}

void Reconstructor::setReconstruction(ndim::pointer<const float, 2> data)
{
	m->m_stepSize = 1.f / m->m_resolutionReconstruction / 10.f;

	assert(data.width() == data.height());
	assert(data.width() == m->m_resolutionReconstruction);
	int resolution = int(data.width());
	assert(data.isContiguous());
	glBindTexture(GL_TEXTURE_2D, m->m_programs.textures[GL::Tex_ReconTest]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, resolution, resolution, GL_RED, GL_FLOAT, data.data);
	glBindTexture(GL_TEXTURE_2D, 0);

	m->project_toSinoTest_fromReconTest();

	acceptTestDirect();

	GL::assert_glError();
}

void Reconstructor::guess()
{
	m->m_stepSize = 1.f / m->m_resolutionReconstruction / 10.f;

	m->guess_toRecon_fromSinoOrg();
	m->mask_toReconTest_fromRecon();
	m->project_toSinoTest_fromReconTest();

	acceptTestDirect();

	GL::assert_glError();
}

bool Reconstructor::step()
{
	m->sum_toReconTest_fromReconAndGradient(m->m_stepSize);
	m->project_toSinoTest_fromReconTest();

	m->likelihood_fromSinoAndSinoTest();
	m->sum_Likelihood();

	if (m->m_likelihoodTest < m->m_likelihood) {
		m->m_stepSize /= 16;
		//		m->accept_test();
		//		m->gradient_FromSinoAndSinoRecon();
		GL::assert_glError();
		return false;
	} else
		m->m_stepSize *= 2;

	m->accept_test();
	m->gradient_FromSinoAndSinoRecon();

	GL::assert_glError();
	return true;
}

void Reconstructor::clear()
{
    m->m_programs.clearTextures();
    m->m_filledSinogram = 0;
    GL::assert_glError();
}

void Reconstructor::readTexture(const ndim::pointer<quint16, 2> data, Reconstructor::ReconTextures texture)
{
	GL::Textures tex = static_cast<GL::Textures>(texture);
	assert(m->m_programs.textureHeight(tex) == data.height());
	assert(m->m_programs.textureWidth(tex) == data.width());
	m->gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m->m_programs.textures[tex], 0);
	assert(m->gl.glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, GLsizei(data.width()), GLsizei(data.height()));

	glReadPixels(0, 0, GLsizei(data.width()), GLsizei(data.height()), GL_RED, GL_UNSIGNED_SHORT, data.data);
	GL::assert_glError();
}

void Reconstructor::readTexture(const ndim::pointer<float, 2> data, Reconstructor::ReconTextures texture)
{
	GL::Textures tex = static_cast<GL::Textures>(texture);
	assert(m->m_programs.textureHeight(tex) == data.height());
	assert(m->m_programs.textureWidth(tex) == data.width());
	m->gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m->m_programs.textures[tex], 0);
	assert(m->gl.glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, GLsizei(data.width()), GLsizei(data.height()));

	glReadPixels(0, 0, GLsizei(data.width()), GLsizei(data.height()), GL_RED, GL_FLOAT, data.data);
	GL::assert_glError();
}

void Reconstructor::readReconstruction(const ndim::pointer<float, 2> data)
{
	assert(data.width() == m->m_resolutionReconstruction && data.height() == m->m_resolutionReconstruction);
	assert(data.isContiguous());

	m->gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m->m_programs.textures[GL::Tex_Recon], 0);
	assert(m->gl.glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, m->m_resolutionReconstruction, m->m_resolutionReconstruction);

	glReadPixels(0, 0, m->m_resolutionReconstruction, m->m_resolutionReconstruction, GL_RED, GL_FLOAT, data.data);
	GL::assert_glError();
}

float Reconstructor::stepSize() const
{
	return m->m_stepSize;
}

void Reconstructor::setStepSize(float value)
{
	m->m_stepSize = value;
}

} // namespace tomo

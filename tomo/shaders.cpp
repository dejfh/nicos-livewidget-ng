#include "tomo/shaders.h"
#include "tomo/reconstructor.h"

#include <string>
#include <vector>

#include "safecast.h"

#include <ndim/layout.h>

namespace tomo
{

namespace GL
{

/*
 * Texutres
 * Intensity: openBeam intensity - may be optimized in later versions
 * Sino: sinogram
 * Recon: reconstructed volume
 * SinoRecon: sinogram of reconstructed volume
 * ReconTest: candidate for next reconstruction
 * SinoReconTest: sinogram of candidate for next reconstruction
 * texLikelihood: quality of reconstruction per pixel
 * texSum: summed quality
 * texGradient: quality gradient of the reconstruction
 *
 * texAngleMatrix: transforms x coord in sinogram and depth coord to coord in reconstruction
 * */

const char *texture_names[Tex_Count] = {"texIntensity", "texSinoOrg", "texAngleMatrix", "texReconTest", "texSinoTest", "texLikelihood", "texSum",
	"texSinoRecon", "texRecon", "texGradient"};

const char *variable_names[Var_Count] = {"step", "sumStep", "center", "sinoRange", "depth", "projection"};

const char code_headerVertex[] =
	"#version 120\r\n"
	//	"precision highp float;"

	"uniform vec2 sinoRange;"

	"uniform float center;"
	"uniform float projection;"
	"uniform sampler1D texAngleMatrix;" // angles of the current sinogram

	"uniform sampler2D texSinoTest;"
	"uniform sampler2D texLikelihood;"

	//	"in vec4 position;"
	//	"out vec4 gl_Position;"
	//	"out vec2 coord0;"

	"void main() {";

const char code_footerVertex[] = "}";

const char code_headerFragment[] =
	"#version 120\r\n"
	//	"precision highp float;"

	"uniform float step;"
	"uniform vec2 sumStep;"
	"uniform float center;"
	"uniform float layer;"
	"uniform float depth;"
	"uniform float projection;"
	"uniform sampler1D texAngleMatrix;" // angles of the current sinogram
	"uniform sampler2D texSinoOrg;"		// original sinogram
	"uniform sampler1D texIntensity;"   // openBeam intensity
	"uniform sampler2D texReconTest;"   // next volume of reconstruction
	"uniform sampler2D texSinoTest;"	// next sinogram of reconstruction
	"uniform sampler2D texLikelihood;"
	"uniform sampler2D texSinoRecon;" // current sinogram of reconstruction
	"uniform sampler2D texRecon;"	 // volume of reconstruction
	"uniform sampler2D texGradient;"  // quality gradient

	//	"in vec2 coord0;"
	//	"out vec3 gl_FragColor;"

	"void main() {";

const char code_footerFragment[] = "}";

/*
 * Vertex Shaders
 * */

const char code_vertex_default[] =
	"gl_TexCoord[0].xy = gl_Vertex.xy;"					 // gl_TexCoord[0]: 0..1
	"gl_Position.xy = vec2(-1, -1) + 2 * gl_Vertex.xy;"; // gl_Position: -1..1

const char code_vertex_matrix[] =
	"gl_TexCoord[0].xy = gl_Vertex.xy;" // gl_TexCoord[0]: 0..1
	"vec2 dir = texture1D(texAngleMatrix, projection).rg;"
	"mat2 matrix = mat2(dir.x, dir.y, -dir.y, dir.x);"
	"vec2 pos = vec2(gl_Vertex.x - center, gl_Vertex.y - .5f);"
	"gl_Position.xy = matrix * (2 * pos);"; // TODO: Passt noch nicht wirklich.

const char code_vertex_sinogram[] =
	"gl_TexCoord[0].xy = gl_Vertex.xy * sinoRange;"		 // gl_TexCoord[0].x: 0..1, gl_TexCoord[0].y: 0..(filled/capacity)
	"gl_Position.xy = vec2(-1, -1) + 2 * gl_Vertex.xy;"; // gl_Position: -1..1

/*
 * Fragment Shaders
 * */

const char code_fragment_default[] = "gl_FragColor.rgb = texture2D(tex0, gl_TexCoord[0].xy).rgb;";

const char code_fragment_guess_ToRecon_FromSinoOrg[] =
	"vec2 pos = vec2(gl_TexCoord[0].x, projection);"
	"gl_FragColor.r = log(texture2D(texSinoOrg, pos).r / texture1D(texIntensity, pos.x).r) / step;";

const char code_fragment_mask_ToReconTest_FromRecon[] =
	"if (distance(gl_TexCoord[0].xy, vec2(.5))<.5)"
	"    gl_FragColor.r = min(0, texture2D(texRecon, gl_TexCoord[0].xy).r);"
	"else"
	"    gl_FragColor.r = 0.f;";

const char code_fragment_project_ToSinoTest_FromReconTest[] =
	"vec2 dir = texture1D(texAngleMatrix, gl_TexCoord[0].y).rg;"
	"mat2 matrix = mat2(dir.x, dir.y, -dir.y, dir.x);"
	"vec2 coord1 = vec2(.5f) + matrix * vec2(gl_TexCoord[0].x - center, depth - .5f);"
	"gl_FragColor.r = texture2D(texReconTest, coord1).r;";

const char code_fragment_likelihood_FromSinoOrgAndSinoTest[] =
	"float sinoTest = texture1D(texIntensity, gl_TexCoord[0].x).r * exp(texture2D(texSinoTest, gl_TexCoord[0].xy).r);"
	"gl_FragColor.r = texture2D(texSinoOrg, gl_TexCoord[0].xy).r * log(sinoTest) - sinoTest;";

const char code_fragment_sumLikelihood[] =
	"vec2 coord3 = gl_TexCoord[0].xy + sumStep;"
	"vec2 coord2 = vec2(gl_TexCoord[0].x, coord3.y);"
	"vec2 coord1 = vec2(coord3.x, gl_TexCoord[0].y);"
	"gl_FragColor.r = texture2D(texLikelihood, gl_TexCoord[0].xy).r + texture2D(texLikelihood, coord1).r + texture2D(texLikelihood, coord2).r + "
	"texture2D(texLikelihood, "
	"coord3).r;";

const char code_fragment_gradient_FromSinoOrgAndSinoRecon[] =
	"vec2 pos = vec2(gl_TexCoord[0].x, projection);"
	"gl_FragColor.r = texture2D(texSinoOrg, pos).r - texture1D(texIntensity, gl_TexCoord[0].x).r * exp(texture2D(texSinoRecon, pos).r);";

const char code_fragment_addGradient_FromReconAndGradient[] =
	"if (distance(gl_TexCoord[0].st, vec2(.5))<=.5)"
	"    gl_FragColor.r = min(0, texture2D(texRecon, vec2(gl_TexCoord[0].st)).r + step * texture2D(texGradient, vec2(gl_TexCoord[0].st)).r);" // TODO:
																																			  // max
																																			  // at 0?
	//	"    gl_FragColor.r = texture2D(texRecon, vec2(gl_TexCoord[0].st)).r + step * texture2D(texGradient, vec2(gl_TexCoord[0].st)).r;"
	"else"
	"    gl_FragColor.r = 0.f;";

TextureTypes textureType(Textures tex)
{
	switch (tex) {
	case Tex_Intensity:
		return TexT_SinoLine;
	case Tex_SinoOrg:
		return TexT_Sino;
	case Tex_AngleMatrix:
		return TexT_SinoColumnMatrix;
	case Tex_ReconTest:
		return TexT_Volume;
	case Tex_SinoTest:
		return TexT_Sino;
	case Tex_Likelihood:
		return TexT_Sino;
	case Tex_Sum:
		return TexT_Sino;
	case Tex_SinoRecon:
		return TexT_Sino;
	case Tex_Recon:
		return TexT_Volume;
	case Tex_Gradient:
		return TexT_Volume;
	default:
		assert(!"Unknown Texture.");
		return TexT_Count;
	}
}

bool is2DTexture(TextureTypes texType)
{
	switch (texType) {
	case TexT_Sino:
	case TexT_Volume:
		return true;
	case TexT_SinoLine:
	case TexT_SinoColumn:
	case TexT_SinoColumnMatrix:
		return false;
	default:
		assert(false);
		return true;
	}
}

bool is2DTexture(Textures tex)
{
	return is2DTexture(textureType(tex));
}

enum GL_ERROR_CODES {
	GEC_NO_ERROR = 0,
	GEC_INVALID_ENUM = 0x0500,
	GEC_INVALID_VALUE = 0x0501,
	GEC_INVALID_OPERATION = 0x0502,
	GEC_STACK_OVERFLOW = 0x0503,
	GEC_STACK_UNDERFLOW = 0x0504,
	GEC_OUT_OF_MEMORY = 0x0505
};

void assert_glError()
{
	int err;
	assert(!(err = glGetError()));
	jfh::unused_variable(err);
}

GLuint createShader(QGLFunctions &gl, const char *code, GLenum type)
{
	GLuint shader = gl.glCreateShader(type);
	const char *parts[3];
	parts[0] = type == GL_VERTEX_SHADER ? code_headerVertex : code_headerFragment;
	parts[1] = code;
	parts[2] = type == GL_VERTEX_SHADER ? code_footerVertex : code_footerFragment;
	gl.glShaderSource(shader, 3, parts, 0);
	gl.glCompileShader(shader);

	GLint result = GL_FALSE;
	gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE) {
		int infoLength;
		gl.glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLength);
		std::string msg(infoLength, 0);
		gl.glGetShaderInfoLog(shader, infoLength, NULL, &msg[0]);
		gl.glDeleteShader(shader);
		assert(!"Error compiling shader program.");
	}
	return shader;
}

GLuint createShaderProgram(QGLFunctions &gl, const char *fragmentCode, const char *vertexCode = 0)
{
	if (vertexCode == 0)
		vertexCode = code_vertex_default;
	GLuint vertexShader = createShader(gl, vertexCode, GL_VERTEX_SHADER);
	GLuint fragmentShader = createShader(gl, fragmentCode, GL_FRAGMENT_SHADER);

	GLuint program = gl.glCreateProgram();
	gl.glAttachShader(program, vertexShader);
	gl.glAttachShader(program, fragmentShader);
	gl.glLinkProgram(program);

	gl.glDeleteShader(vertexShader);
	gl.glDeleteShader(fragmentShader);

	GLint result = GL_FALSE;
	gl.glGetProgramiv(program, GL_LINK_STATUS, &result);
	if (result == GL_FALSE) {
		int infoLength;
		gl.glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLength);
		std::string msg(infoLength, 0);
		gl.glGetProgramInfoLog(program, infoLength, NULL, &msg[0]);
		gl.glDeleteProgram(program);
		program = 0;
		assert(!"Error linking shader program.");
	}
	return program;
}

void Program::init(ProgramGroup &g, Programs program)
{
	assert_glError();
	switch (program) {
	case ProgGuess_ToRecon_FromSinoOrg:
		blend = true;
		target = Tex_Recon;
		this->program = createShaderProgram(g.gl, code_fragment_guess_ToRecon_FromSinoOrg, code_vertex_matrix);
		break;
	case ProgMask_ToReconTest_FromRecon:
		blend = false;
		target = Tex_ReconTest;
		this->program = createShaderProgram(g.gl, code_fragment_mask_ToReconTest_FromRecon);
		break;
	case ProgProject_ToSinoTest_FromReconTest:
		blend = true;
		target = Tex_SinoTest;
		this->program = createShaderProgram(g.gl, code_fragment_project_ToSinoTest_FromReconTest, code_vertex_sinogram);
		break;
	case ProgLikelihood_FromSinoOrgAndSinoTest:
		blend = false;
		target = Tex_Likelihood;
		this->program = createShaderProgram(g.gl, code_fragment_likelihood_FromSinoOrgAndSinoTest, code_vertex_sinogram);
		break;
	case ProgSum_Likelihood:
		blend = false;
		target = Tex_Sum;
		this->program = createShaderProgram(g.gl, code_fragment_sumLikelihood, code_vertex_sinogram);
		break;
	case ProgGradient_FromSinoOrgAndSinoRecon:
		blend = true;
		target = Tex_Gradient;
		this->program = createShaderProgram(g.gl, code_fragment_gradient_FromSinoOrgAndSinoRecon, code_vertex_matrix);
		break;
	case ProgSum_ToReconTest_FromReconAndGradient:
		blend = false;
		target = Tex_ReconTest;
		this->program = createShaderProgram(g.gl, code_fragment_addGradient_FromReconAndGradient);
		break;
	default:
		assert(false);
	}
	assert_glError();
	for (int i = 0; i < Var_Count; ++i) {
		loc_var[i] = g.gl.glGetUniformLocation(this->program, variable_names[i]);
		assert_glError();
	}
	for (int i = 0; i < Tex_Count; ++i) {
		loc_tex[i] = g.gl.glGetUniformLocation(this->program, texture_names[i]);
		assert_glError();
	}
	assert_glError();
}

void Program::use(ProgramGroup &g)
{
	assert_glError();
	QGLFunctions &gl = g.gl;
	gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.textures[target], 0);
	assert(gl.glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	assert_glError();
	glViewport(0, 0, g.textureWidth(target), g.targetHeight(target));
	assert_glError();
	gl.glUseProgram(program);
	assert_glError();
	int texUnit = 0;
	for (int i = 0; i < Tex_Count; ++i) {
		Textures texture(static_cast<Textures>(i));
		useTexture(gl, texUnit, g.textures[texture], loc_tex[texture], is2DTexture(texture) ? GL_TEXTURE_2D : GL_TEXTURE_1D);
		assert_glError();
	}
	if (blend) {
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
	} else
		glDisable(GL_BLEND);

	assert_glError();
}

void Program::useTexture(QGLFunctions &gl, int &texUnit, GLuint tex, GLint loc, GLenum texType)
{
	assert_glError();
	if (loc < 0)
		return;
	gl.glActiveTexture(GL_TEXTURE0 + texUnit);
	assert_glError();
	glBindTexture(texType, tex);
	assert_glError();
	gl.glUniform1i(loc, texUnit);
	assert_glError();
	++texUnit;
	assert_glError();
}

unsigned int ProgramGroup::textureWidth(TextureTypes texType)
{
	return tex_width[texType];
}

unsigned int ProgramGroup::textureWidth(Textures tex)
{
	return textureWidth(textureType(tex));
}

unsigned int ProgramGroup::textureHeight(TextureTypes texType)
{
	return tex_height[texType];
}

unsigned int ProgramGroup::textureHeight(Textures tex)
{
	return textureHeight(textureType(tex));
}

unsigned int ProgramGroup::targetHeight(TextureTypes texType)
{
	if (texType == TexT_Sino)
		return sino_filled;
	else
		return textureHeight(texType);
}

unsigned int ProgramGroup::targetHeight(Textures tex)
{
	return targetHeight(textureType(tex));
}

const size_t segments = 4;

ProgramGroup::ProgramGroup(QGLFunctions &gl)
	: gl(gl)
	, framebuffer(0)
	, vertexBuffer(0)
{
	textures.fill(0);
	assert_glError();
	{
		gl.glGenFramebuffers(1, &framebuffer);
		gl.glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		assert_glError();
	}
	{
		gl.glGenBuffers(1, &vertexBuffer);
		gl.glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		std::array<float, (segments + 1) * (segments + 1) * 2> vertexBufferData;
		auto it = vertexBufferData.begin();
		for (float y = 0; y <= 1; y += 1.f / segments)
			for (float x = 0; x <= 1; x += 1.f / segments) {
				*it++ = x;
				*it++ = y;
			}
		gl.glBufferData(GL_ARRAY_BUFFER, vertexBufferData.size() * sizeof(float), vertexBufferData.data(), GL_STATIC_DRAW);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(2, GL_FLOAT, 0, 0);
		gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
		assert_glError();
	}
	{
		gl.glGenBuffers(1, &indexBuffer);
		gl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
		ndim::layout<2> layout(ndim::makeSizes(segments + 1, segments + 1), hlp::byte_offset_t(1));
		std::array<GLubyte, segments * segments * 6> indexBufferData;
		auto it = indexBufferData.begin();
		for (size_t y = 0; y < segments; ++y)
			for (size_t x = 0; x < segments; ++x) {
				*it++ = GLubyte(layout.offsetOf(x, y).value);
				*it++ = GLubyte(layout.offsetOf(x + 1, y).value);
				*it++ = GLubyte(layout.offsetOf(x, y + 1).value);
				*it++ = GLubyte(layout.offsetOf(x + 1, y).value);
				*it++ = GLubyte(layout.offsetOf(x + 1, y + 1).value);
				*it++ = GLubyte(layout.offsetOf(x, y + 1).value);
			}
		gl.glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferData.size() * sizeof(GLubyte), indexBufferData.data(), GL_STATIC_DRAW);
	}

	glClearColor(0, 0, 0, 0);
	glBlendFunc(GL_ONE, GL_ONE);
}

ProgramGroup::~ProgramGroup()
{
}

void ProgramGroup::draw()
{
	const size_t vertices = segments * segments * 6;
	for (size_t index = 0; index < vertices; index += 3)
		glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, reinterpret_cast<void *>(index));
	GL::assert_glError();
}

Program &ProgramGroup::use(Programs program)
{
	programs[program].use(*this);
	return programs[program];
}

void ProgramGroup::createTextures(TextureTypes type, int width, int height)
{
	bool is2D = is2DTexture(type);
	tex_width[type] = width;
	tex_height[type] = is2D ? height : 1;
	for (int i = 0; i < Tex_Count; ++i)
		if (textureType(static_cast<Textures>(i)) == type) {
			GLuint &texture = textures[i];
			assert_glError();
			if (!texture)
				glGenTextures(1, &texture);
			assert_glError();
			GLenum tex_type = is2D ? GL_TEXTURE_2D : GL_TEXTURE_1D;
			glBindTexture(tex_type, texture);
			assert_glError();
			if (type == TexT_SinoColumnMatrix)
				glTexImage1D(GL_TEXTURE_1D, 0, GL_RG32F, width, 0, GL_RG, GL_FLOAT, 0);
			else if (is2D)
				glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
			else
				glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, width, 0, GL_RED, GL_FLOAT, 0);
			assert_glError();
			glTexParameterf(tex_type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			assert_glError();
			if (is2D)
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			assert_glError();
			glTexParameteri(tex_type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			assert_glError();
			glTexParameteri(tex_type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			assert_glError();
			glTexParameteri(tex_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			assert_glError();
			glTexParameteri(tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			assert_glError();

			glBindTexture(tex_type, 0);
			assert_glError();
		}
}

void ProgramGroup::clearTextures()
{
	glDeleteTextures(Tex_Count, textures.data());
	textures.fill(0);
}

void ProgramGroup::createPrograms()
{
	for (int i = 0; i < Prog_Count; ++i) {
		Program &program = programs[i];
		program.init(*this, static_cast<Programs>(i));
	}
	assert_glError();
}

} // namespace GL

} // namespace tomo

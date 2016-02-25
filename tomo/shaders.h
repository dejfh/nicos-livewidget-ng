#ifndef TOMO_SHADERS_H
#define TOMO_SHADERS_H

#include <array>
#include <map>

#include <QGLFunctions>
#include <QGLBuffer>
#include <QGLFramebufferObject>

#include <GL/gl.h>
#include <GL/glext.h>

namespace tomo
{

namespace GL
{

class Reconstructor;

enum TextureTypes { TexT_Sino, TexT_Volume, TexT_SinoLine, TexT_SinoColumn, TexT_SinoColumnMatrix, TexT_Count };

enum Textures {
	Tex_Intensity,
	Tex_SinoOrg,
	Tex_AngleMatrix,
	Tex_ReconTest,
	Tex_SinoTest,
	Tex_Likelihood,
	Tex_Sum,
	Tex_SinoRecon,
	Tex_Recon,
	Tex_Gradient,
	Tex_Count
};
enum Programs {
	ProgGuess_ToRecon_FromSinoOrg,
	ProgMask_ToReconTest_FromRecon,
	ProgProject_ToSinoTest_FromReconTest,
	ProgLikelihood_FromSinoOrgAndSinoTest,
	ProgSum_Likelihood,
	ProgGradient_FromSinoOrgAndSinoRecon,
	ProgSum_ToReconTest_FromReconAndGradient,
	Prog_Count
};
enum Variables { Var_Step, Var_SumStep, Var_Center, Var_SinoRange, Var_Depth, Var_Projection, Var_Count };

TextureTypes textureType(Textures tex);
bool is2DTexture(TextureTypes texType);
bool is2DTexture(Textures tex);
void assert_glError();

struct ProgramGroup;

struct Program {
	GLuint program;
	bool blend;
	Textures target;

	std::array<GLint, Var_Count> loc_var;
	std::array<GLint, Tex_Count> loc_tex;

	void init(ProgramGroup &g, Programs program);

	void use(ProgramGroup &g);

private:
	void useTexture(QGLFunctions &gl, int &texUnit, GLuint tex, GLint loc, GLenum texType);
};

struct ProgramGroup {
	friend struct Program;

	QGLFunctions &gl;
	GLenum framebuffer;
	GLenum vertexBuffer;
	GLenum indexBuffer;

	std::array<int, TexT_Count> tex_width;
	std::array<int, TexT_Count> tex_height;

	int sino_filled;

	unsigned int textureWidth(TextureTypes texType);
	unsigned int textureWidth(Textures tex);
	unsigned int textureHeight(TextureTypes texType);
	unsigned int textureHeight(Textures tex);
	unsigned int targetHeight(TextureTypes texType);
	unsigned int targetHeight(Textures tex);

	ProgramGroup(QGLFunctions &gl);
	~ProgramGroup();

	std::array<GLuint, Tex_Count> textures;
	std::array<Program, Prog_Count> programs;

	void draw();
	Program &use(Programs program);

	void createTextures(TextureTypes type, int width, int height);
	void clearTextures();

	void createPrograms();
};

} // namespace GL

} // namespace tomo

#endif // TOMO_SHADERS_H

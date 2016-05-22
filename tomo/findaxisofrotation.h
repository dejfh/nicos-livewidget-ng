#ifndef TOMO_FINDAXISOFROTATION_H
#define TOMO_FINDAXISOFROTATION_H

#include <cstddef>

#include <helper/asyncprogress.h>

#include "ndim/pointer.h"

#include "tomo/axisofrotation.h"

namespace tomo
{

/*!
 \brief Uses two opposing images to find the axis of rotation.
*/
template <typename _T>
class FindAxisOfRotation
{
public:
	/*!
	 \brief Creates a new instance of FindRotationAxis.

	 \param img1 The front image
	 \param img2 The back image
	*/
	FindAxisOfRotation(ndim::pointer<const _T, 2> img1, ndim::pointer<const _T, 2> img2);

public:
	const ndim::pointer<const _T, 2> img1; /*!< front image */
	const ndim::pointer<const _T, 2> img2; /*!< back image */

	AxisOfRotation axis;

	//	double center;   /*!< x-coordinate of the axis of rotation at the edge y=0.0 */
	//	double tanAngle; /*!< Tangens of the slope of the axis of rotation */

public:
	/*!
	 \brief Searches the axis of rotation.

	 Starts the search with an initial \ref guess, then continues with \ref runAsChild.
	 At last the axis of rotation is \ref optimize "optimized" with smaller step sizes.

	 \param callback Callback, which is called from time to time during the search
	*/
	void run(hlp::AsyncPartialProgress progress);

protected:
	/*!
	 \brief Searches the axis of rotation.

	 The images are binned multiple times, until the image size is under a threshold.
	 The parameters for the axis of rotation are \ref optimize "optimized",
	 transferred to the next less binned version, optimized, transferred and so on.

	 \param callback Callback, which is called from time to time during the search, or NULL
	 \param obj A custom parameter forwarded to the callback
	*/
	void runStep(hlp::AsyncPartialProgress progress);

	/*!
	 \brief Does an initial guess for the axis of rotation.

	 Uses the center of intensity of the whole image and of one line for the initial guess.
	*/
	void guess();

	/*!
	 \brief Calculates the deviation per pixel between the opposing images for a given axis of rotation.

	 \param center The x-coordinate of the axis of rotation at y = 0
	 \param tanAngle The tangens of the slope of the axis of rotation
	 \return double The deviation per pixel betwenn the opposing images
	*/
	double calcDeviation(double center, double tanAngle) const;

	/*!
	 \brief Optimizes the axis of rotation.

	 The deviation per pixel is calculated for small variations of the axis of rotation.
	 The axis of rotation is set to the best results.
	 The amount of configurations tested is (2 * stepCount + 1)^2.

	 \param stepCenter The step size to vary the position of the axis of rotation
	 \param stepAngle The step size to vary the slope of the axis of rotation
	 \param stepCount The amount of steps to vary in every direction
	*/
	void optimize(hlp::AsyncPartialProgress progress, double stepSize, int stepCount);

	/*!
	 \brief Calculates the center of intensity for a specific region of an image.

	 \param img The image
	 \param fromX The horizontal lower (including) bound of the region
	 \param toX The horizontal upper (excluding) bound of the region
	 \param fromY The vertical lower (including) bound of the region
	 \param toY The vertical upper (excluding) bound of the region
	 \param centerX Receives the x-coordinate of the center of intensity
	 \param centerY Receives the y-coordinate of the center of intensity
	 \return double The summed intensity of the region
	*/
	double findCenterOfIntensity(ndim::pointer<const _T, 2> img, double *centerX, double *centerY);

public:
	/*!
	 \brief Fills an image with the flipped opposing image.

	 Fills an image with the flipped version of the opposing (second) image.
	 The same width, stride and height as for the original image are assumed.

	 \param img The image to be filled
	*/
	void fillWithFlippedBack(ndim::pointer<_T, 2> out) const;

	/*!
	 \brief Gets a single layer of an image, perpendicular to the axis of rotation

	 \param output Receives the layer
	 \param size The maximum size of the layer
	 \param axis_x Receives the coordinate of the axis of rotation within the selected layer
	 \param img The image to take the layer from
	 \param layer The number of the layer to get from the image
	*/
	void getLayerOfImage(ndim::pointer<const _T, 2> in, ndim::pointer<_T, 1> out, size_t layer, double *axis_x) const;
};

} // namespace tomo

#endif // TOMO_FINDAXISOFROTATION_H

#include "tomo/findaxisofrotation.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

#include "helper/helper.h"

#include "ndim/pointer.h"
#include "ndim/algorithm_omp.h"

using namespace std;

using ndim::pointer;

namespace tomo
{

template <typename _T>
FindAxisOfRotation<_T>::FindAxisOfRotation(ndim::pointer<const _T, 2> img1, ndim::pointer<const _T, 2> img2)
	: img1(img1)
	, img2(img2)
{
	assert(img1.width() == img2.width() && img1.height() == img2.height());
}

template <typename _T>
void FindAxisOfRotation<_T>::run(hlp::AsyncPartialProgress progress)
{
	progress.throwIfCancelled();
	guess();
	progress.throwIfCancelled();
	runStep(progress.sub(.5));
	double stepSize = .5;
	for (int i = 0; i < 3; i++) {
		progress.throwIfCancelled();
		optimize(progress.sub(.5f / 3), stepSize, 1);
		stepSize *= .5;
	}
}

template <typename _T_in, typename _T_out>
void shrink(ndim::pointer<_T_in, 2> in, ndim::pointer<_T_out, 2> out)
{
	_T_out *pout = out.data;
	const _T_in *p1 = in.data;
	const _T_in *p2 = p1 + in.byte_strides[0];
	const _T_in *p3 = p1 + in.byte_strides[1];
	const _T_in *p4 = p1 + in.byte_strides[0] + in.byte_strides[1];
	in.sizes[0] /= 2;
	in.sizes[1] /= 2;
	in.byte_strides[0] *= 2;
	in.byte_strides[1] *= 2;
	assert(out.width() == in.width() && out.height() == in.height());
	const array<size_t, 2> &sizes(in.sizes);
	const array<hlp::byte_offset_t, 2> hops_in(in.getHops());
	const array<hlp::byte_offset_t, 2> hops_out(out.getHops());
	for (size_t y = sizes[1]; y > 0; --y) {
		for (size_t x = sizes[0]; x > 0; --x) {
			*pout = (*p1 + *p2 + *p3 + *p4) / 4;
			pout += hops_out[0];
			p1 += hops_in[0];
			p2 += hops_in[0];
			p3 += hops_in[0];
			p4 += hops_in[0];
		}
		pout += hops_out[1];
		p1 += hops_in[1];
		p2 += hops_in[1];
		p3 += hops_in[1];
		p4 += hops_in[1];
	}
}

template <typename _T>
void FindAxisOfRotation<_T>::runStep(hlp::AsyncPartialProgress progress)
{
	progress.throwIfCancelled();
	if (std::min(img1.width(), img1.height()) > 32) {
		size_t widthS = img1.width() / 2;
		size_t heightS = img1.height() / 2;
		vector<_T> va1s(widthS * heightS);
		vector<_T> va2s(widthS * heightS);
		ndim::pointer<_T, 2> img1s = ndim::make_ptr_contiguous(va1s.data(), widthS, heightS);
		ndim::pointer<_T, 2> img2s = ndim::make_ptr_contiguous(va2s.data(), widthS, heightS);

		shrink(img1, img1s);
		shrink(img2, img2s);

		progress.advance(.05f);
		FindAxisOfRotation<_T> child(img1s, img2s);
		child.axis = AxisOfRotation::fromEdge(widthS, heightS, axis.axisXOfEdge() / 2, axis.tan);
		child.runStep(progress.sub(.25));
		progress.throwIfCancelled();
		axis = AxisOfRotation::fromEdge(axis.width, axis.height, child.axis.axisXOfEdge() * 2, axis.tan);
		optimize(progress.sub(.7f), 1.f, 2);
	} else
		optimize(std::move(progress), 1.f, 4);
}

template <typename _T>
void FindAxisOfRotation<_T>::guess()
{
	double x1, y1, x2, y2;
	findCenterOfIntensity(img1, &x1, &y1);
	findCenterOfIntensity(img2, &x2, &y2);
	double centerX = (x1 + x2) / 2.f;
	double centerY = (y1 + y2) / 2.f;
	double x1a, y1a, x2a, y2a;
	size_t y = img1.height() / 4;
	ndim::pointer<const _T, 2> sub_img1(img1);
	sub_img1.selectRange(1, y, y + 1);
	ndim::pointer<const _T, 2> sub_img2(img2);
	sub_img2.selectRange(1, y, y + 1);
	findCenterOfIntensity(sub_img1, &x1a, &y1a);
	findCenterOfIntensity(sub_img2, &x2a, &y2a);
	double centerXa = (x1a + x2a) / 2.f;
	double centerYa = (y1a + y2a) / 2.f + y;

	double tan = -(centerX - centerXa) / (centerY - centerYa);
	double center_edge = centerX + tan * centerY;
	axis = AxisOfRotation::fromEdge(img1.width(), img1.height(), center_edge, tan);
}

// Returns x².
template <typename _T>
inline _T sq(_T x)
{
	return x * x;
}

template <typename _T>
double FindAxisOfRotation<_T>::calcDeviation(double center, double tanAngle) const
{
	const size_t width = img1.width();
	const size_t height = img1.height();
	const AxisOfRotation axis = AxisOfRotation::fromEdge(width, height, center, tanAngle);

	double diff = 0.f;
	size_t count = 0;

	double x_axis = axis.center_layer_0;
	double y_axis = axis.yOfLayer(x_axis, 0);
	for (; y_axis < height; x_axis -= axis.sin_times_cos, y_axis += axis.cos_square) {
		int x1 = static_cast<int>(x_axis - .25f);
		int x2 = static_cast<int>(x_axis + .25f);
		double y1 = y_axis + (x1 - x_axis) * axis.tan;
		double y2 = y_axis + (x2 - x_axis) * axis.tan;

		for (;; x1--, x2++, y1 -= axis.tan, y2 += axis.tan) {
			if (x1 < 0 || x2 < 0)
				break;
			if (size_t(x1) >= width || size_t(x2) >= width)
				break;
			if (y1 >= height || y2 >= height)
				break;
			if (y1 < 0 || y2 < 0)
				break;

			diff += sq(img1(x1, size_t(y1)) - img2(x2, size_t(y2)));
			diff += sq(img1(x2, size_t(y2)) - img2(x1, size_t(y1)));
			++count;
		}
	}
	return sqrt(diff) / count;
}

template <typename _T>
void FindAxisOfRotation<_T>::fillWithFlippedBack(ndim::pointer<_T, 2> out) const
{
	const size_t width = img2.width();
	const size_t height = img2.height();

	assert(out.width() == width && out.height() == height);

	double x_axis = axis.center_layer_0;
	double y_axis = axis.yOfLayer(x_axis, 0);
	for (; y_axis < height; x_axis -= axis.sin_times_cos, y_axis += axis.cos_square) {
		int x1 = int(x_axis - .25f);
		int x2 = int(x_axis + .25f);
		double y1 = y_axis + (x1 - x_axis) * axis.tan;
		double y2 = y_axis + (x2 - x_axis) * axis.tan;

		for (;; x1--, x2++, y1 -= axis.tan, y2 += axis.tan) {
			if (x1 < 0 || x2 < 0)
				break;
			if (size_t(x1) >= width || size_t(x2) >= width)
				break;
			if (y1 >= height || y2 >= height)
				break;
			if (y1 < 0 || y2 < 0)
				break;

			out(x1, size_t(y1)) = img2(x2, size_t(y2));
			out(x2, size_t(y2)) = img2(x1, size_t(y1));
		}
	}
}

template <typename _T>
void FindAxisOfRotation<_T>::optimize(hlp::AsyncPartialProgress progress, double stepSize, int stepCount)
{
	double stepAngle = stepSize * 2 / std::max(img1.width(), img1.height());
	double center = axis.axisXOfEdge();
	double tan = axis.tan;
	double q = std::numeric_limits<double>::max();
	double amount = 1.0 / sq(stepCount * 2 + 1);
	for (int i = -stepCount; i <= stepCount; i++)
		for (int j = -stepCount; j <= stepCount; j++) {
			double q_new = calcDeviation(center + i * stepSize + j * stepSize, tan + j * stepAngle);
			if (q_new < q) {
				q = q_new;
				axis = AxisOfRotation::fromEdge(axis.width, axis.height, center, tan);
			}
			progress.advance(amount);
			progress.throwIfCancelled();
		}
}

template <typename _T>
double FindAxisOfRotation<_T>::findCenterOfIntensity(ndim::pointer<const _T, 2> img, double *centerX, double *centerY)
{
	struct accumulate_t {
		double weight, x, y;
		accumulate_t()
			: weight(0)
			, x(0)
			, y(0)
		{
		}
		accumulate_t(double weight, double x, double y)
			: weight(weight)
			, x(x)
			, y(y)
		{
		}
		accumulate_t operator+(accumulate_t other) const
		{
			other.weight += this->weight;
			other.x += this->x;
			other.y += this->y;
			return other;
		}
	};

	auto operation = [](accumulate_t acc, std::array<size_t, 2> coords, _T value) {
		double weight(value);
		return acc + (accumulate_t(weight, weight * coords[0], weight * coords[1]));
	};

	accumulate_t accumulated;
#pragma omp parallel
	{
		accumulate_t t = ndim::accumulate_omp(operation, accumulate_t(), img);
#pragma omp critical
		accumulated = accumulated + t;
	}

	*centerX = accumulated.x / accumulated.weight;
	*centerY = accumulated.y / accumulated.weight;
	return accumulated.weight;
}

template <typename _T>
void FindAxisOfRotation<_T>::getLayerOfImage(ndim::pointer<const _T, 2> in, ndim::pointer<_T, 1> out, size_t layer, double *axis_x) const
{
	size_t width = in.width();
	size_t height = in.height();
	assert(out.width() == width);
	double y = axis.yOfLayer(.5, layer);
	for (size_t x = 0; x < width; x++) {
		if (y < 0 || y >= height)
			out(x) = 0;
		else
			out(x) = in(x, size_t(y));
		y += axis.tan;
	}
	if (axis_x)
		*axis_x = axis.axisXOfLayer(layer);
}

template class FindAxisOfRotation<signed char>;
template class FindAxisOfRotation<signed short>;
template class FindAxisOfRotation<signed long>;
template class FindAxisOfRotation<signed long long>;
template class FindAxisOfRotation<unsigned char>;
template class FindAxisOfRotation<unsigned short>;
template class FindAxisOfRotation<unsigned long>;
template class FindAxisOfRotation<unsigned long long>;
template class FindAxisOfRotation<float>;
template class FindAxisOfRotation<double>;

} // namespace tomo

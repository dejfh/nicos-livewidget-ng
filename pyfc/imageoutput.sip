class ImageOutputChain {
%TypeHeaderCode
#include "fc/chains/imageoutput.h"
using ImageOutputChain = fc::chains::ImageOutput;
%End
	private:
	ImageOutputChain(const ImageOutputChain& ); // Disable allocation as array
public:
	ImageOutputChain();

	void setSource(Filter2d source);

	void setLog(bool log);

	void setColor(bool color);
	void setInvert(bool invert);

	void setColormapRange(double a, double b);
	%MethodCode
		sipCpp->setColormapRange(std::make_pair(a0, a1));
	%End
	void setAutoColormapRange(bool fullRange);

	void resetRegionOfInterest();
	void setRegionOfInterest(long long x, long long y, long long width, long long height);
	%MethodCode
		ndim::range<2> region(ndim::makeIndices(a0, a1), ndim::makeSizes(a2, a3));
		sipCpp->setRegionOfInterest(region);
	%End

	Filter2d roiFilter() const;
	%MethodCode
		sipRes = Filter2d::create(sipCpp->roiFilter());
	%End

	StatisticBuffer statistic() const;
	ImageBuffer image() const;
};

class Filter1d
{
%TypeHeaderCode
#include <memory>
#include <fc/datafilter.h>
using Filter1d = std::shared_ptr<const fc::DataFilter<float, 1>>;
%End
};
class Filter2d
{
%TypeHeaderCode
#include <memory>
#include <fc/datafilter.h>
using Filter2d = std::shared_ptr<const fc::DataFilter<float, 2>>;
%End
};
class Filter3d
{
%TypeHeaderCode
#include <memory>
#include <fc/datafilter.h>
using Filter3d = std::shared_ptr<const fc::DataFilter<float, 3>>;
%End
};
class Filter4d
{
%TypeHeaderCode
#include <memory>
#include <fc/datafilter.h>
using Filter4d = std::shared_ptr<const fc::DataFilter<float, 4>>;
%End
};

class FilterVar
{
%TypeHeaderCode
#include <memory>
#include <fc/datafilter.h>
using FilterVar = std::shared_ptr<const fc::DataFilterVar<float>>;
%End

public:
	FilterVar();
	FilterVar(const Filter1d &);
	FilterVar(const Filter2d &);
	FilterVar(const Filter3d &);
	FilterVar(const Filter4d &);
	FilterVar(const Buffer1d &);
	FilterVar(const Buffer2d &);
	FilterVar(const Buffer3d &);
	FilterVar(const Buffer4d &);
};

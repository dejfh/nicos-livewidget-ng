#ifndef HELPER_QT_VECTOR_H
#define HELPER_QT_VECTOR_H

#include <QVector>

namespace hlp
{

template <typename _ListItem>
void addToQVector(QVector<_ListItem> &)
{
}
template <typename _ListItem, typename _Item0, typename... _ItemN>
void addToQVector(QVector<_ListItem> &list, _Item0 &&item0, _ItemN &&... itemN)
{
	list.append(std::forward<_Item0>(item0));
	addToQVector(list, std::forward<_ItemN>(itemN)...);
}

template <typename _ListItem, typename... _Items>
QVector<_ListItem> makeQVector(_Items &&... items)
{
	QVector<_ListItem> vector;
	vector.reserve(sizeof...(_Items));
	addToQVector(vector, std::forward<_Items>(items)...);
	return vector;
}

} // namespace hlp

#endif // HELPER_QT_VECTOR_H

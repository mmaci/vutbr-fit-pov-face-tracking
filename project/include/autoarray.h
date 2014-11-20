//-----------------------------------------------------------------------------
// autoarray.h
// Macros for small temporary arrays, uses C99 arrays or std::vector
// Jiri Havel, DCGM FIT BUT, Brno
// $Id: autoarray.h 86 2009-04-21 07:13:21Z ihavel $
//-----------------------------------------------------------------------------
#ifndef _AUTO_ARRAY_H_
#define _AUTO_ARRAY_H_

#if defined(__GNUC__)
#define VARIABLE_LENGTH_ARRAYS_SUPPORTED
#endif

#if defined(VARIABLE_LENGTH_ARRAYS_SUPPORTED)//C99 compiler

#define AUTO_ARRAY(type, name, size) type name[size]
#define AUTO_ARRAY_SIZE(array) (sizeof(array)/sizeof(array[0]))
#define AUTO_ARRAY_PTR(array) (array)

#elif defined(__cplusplus)//Non C99 C++ compiler

#include <vector>

#define AUTO_ARRAY(type, name, size) std::vector<type> name(size);
#define AUTO_ARRAY_SIZE(array) (array.size())
#define AUTO_ARRAY_PTR(array) (&array[0])

#else// Other cases

#error "Auto array needs either C99 or C++"

#endif

#endif//_AUTO_ARRAY_H_

#ifndef __QALLOC_H__
#define __QALLOC_H__

#include <stdlib.h>
#include "type.h"

void qinit( size_t n );
void *qalloc( void );
void qfree( void *mem_block );

#endif

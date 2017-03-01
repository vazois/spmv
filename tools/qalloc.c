#include "qalloc.h"

#define MIN_BLOCK_SIZE sizeof( void * )

// We will allocate 1 KiB
#define MEM_CAPACITY (1 << 10)
static U8 mem_array[MEM_CAPACITY];

// Because we are only declaring one static variable (local to this file) to
// be of this type, we do not have to give the structure a name.
//  - this is called an 'unnamed struct'
// This structure is initalized so that mem.size == 0 and mem.top == NULL

static struct {
	U16 size;
	void *top;
} mem = {0, NULL};

/******************************************************************************
* void qinit( size_t n )
*
* This initializes the dynamic allocating data structure.
*  - mem.size is initialized to 0
*  - the minimum size of a block is the size of a pointer.
*  - mem.size is set to either the maximum of argument and the pointer size.
*  - once the data structure has been initialized, do not reinitialize.
*
* During initialization:
*  - set the top of the stack to the base address of mem_array
*  - cast each block of size mem.size as a pointer and assign to it
*    the address of the next block,
*    except for the last block, which is assigned NULL.
******************************************************************************/

void qinit( size_t n ) {
	U16 i, capacity;
	U8 *ptr;

	// Do not allow multiple initializations

	if ( mem.size == 0 ) {
		mem.size = ( n >= MIN_BLOCK_SIZE ) ? n : MIN_BLOCK_SIZE;

		capacity = MEM_CAPACITY/mem.size;

		mem.top = (void *)mem_array;

		ptr = (void *) mem_array;

		for ( i = 0; i < capacity - 1; ++i ) {
			*(void **)ptr = ptr + mem.size;
			ptr = *(void **)ptr;
		}

		*(void **)ptr = NULL;
	}
}

/******************************************************************************
* void *qalloc( void )
*
* Allocate a block of memory.
*  - if the data structure is not initalized or the stack is empty,
*      return NULL
*  - otherwise, pop the top of the stack and return the previous top
*******************************************************************************/

void *qalloc( void ) {
	void *mem_block;

	if ( mem.size == 0 || mem.top == NULL ) {
		return NULL;
	} else {
		mem_block = (void *) mem.top;
		mem.top = *(void **)mem.top;

		return mem_block;
	}
}

/******************************************************************************
* void free( void *mem_block )
*
* Allocate a block of memory.
*  - if the data structure is not initalized or the argument is NULL,
*      do nothing
*  - otherwise, push the argument back onto the stack.
*******************************************************************************/

void qfree( void *mem_block ) {
	if ( mem.size != 0 && mem_block != NULL ) {
		*(void **)mem_block = mem.top;
		mem.top = mem_block;
	}
}

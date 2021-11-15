
/*
           An implementation of top-down splaying with sizes
             D. Sleator <sleator@cs.cmu.edu>, January 1994.
Modified a little by Qingpeng Niu for tracing the global chunck library memory use. Just add a compute sum of size from search node to the right most node.
*/
#ifndef _splay_h
#define _splay_h
#include <stdio.h>
#include <stdlib.h>
//#pragma warning(disable:593)
typedef struct tree_node Tree;
typedef int T;
struct tree_node {
    Tree * left, * right;
    T key;
    T size;   /* maintained to be the number of nodes rooted here */
};

#define compare(i,j) ((i)-(j))
/* This is the comparison.                                       */
/* Returns <0 if i<j, =0 if i=j, and >0 if i>j                   */
 
#define node_size(x) (((x)==NULL) ? 0 : ((x)->size))
/* This macro returns the size of a node.  Unlike "x->size",     */
/* it works even if x=NULL.  The test could be avoided by using  */
/* a special version of NULL which was a real node with size 0.  */
 
Tree * splay (T i, Tree *t);
Tree * insert(T i, Tree * t); 
Tree * delete(T i, Tree *t); 
Tree *find_rank(T r, Tree *t); 
void printtree(Tree * t, int d); 
void freetree(Tree* t);
#endif 


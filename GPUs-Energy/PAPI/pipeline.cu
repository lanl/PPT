/*
** Author(s)      :  Yehia Arafa (yarafa@nmsu.edu) and Ammar ElWazir (ammarwa@nmsu.edu)
** 
** File           :  pipeline.cu  
** 
** Description    :  Host (CPU) code to call each device (GPU) microbenchmark and compute
**                   thier energy using PAPI API - NVML
** 
** Paper          :  Y. Arafa et al., "Verified Instruction-Level Energy Consumption 
**                                     Measurement for NVIDIA GPUs," CF'20
** 
** Notes          :  This code uses some of the open source code provided by PAPI-API               
*/

#include <cuda.h>
#include <stdio.h>
#include "device_functions.cu"
#include "papi.h"

 
#define NUM_EVENTS 1
#define PAPI
 
 // Host function
int main(int argc, const char* argv[]){

int n=10;
/* Host variable Declaration */
int *c;    
/* Device variable Declaration */
int  *d_c;

/* Blocks and Grids*/
dim3 Db = dim3(1);
dim3 Dg = dim3(1,1,1); 

/* Allocation of Device Variables */ 
cudaMalloc((void **)&d_c, n *sizeof(int));
/* Allocation of Host Variables */
c = (int *)malloc(n * sizeof(int)); 

#ifdef PAPI
    int retval, i;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    /* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT 
        FOR THE CUDA DEVICE YOU ARE RUNNING ON.
        RUN papi_native_avail to get a list of CUDA events that are 
        supported on your machine */
        // e.g. on a P100 nvml:::Tesla_P100-SXM2-16GB:power
    char anEvent[64] = "nvml:::GeForce_GTX_TITAN_X:device_0:power";
    char *EventName[] = { anEvent };
    int events[NUM_EVENTS];
    int eventCount = 0;
     
    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT )
        fprintf( stderr, "PAPI_library_init failed\n" );
     
    printf( "PAPI_VERSION     : %4d %6d %7d\n",
            PAPI_VERSION_MAJOR( PAPI_VERSION ),
            PAPI_VERSION_MINOR( PAPI_VERSION ),
            PAPI_VERSION_REVISION( PAPI_VERSION ) );
     
     /* convert PAPI native events to PAPI code */
    for( i = 0; i < NUM_EVENTS; i++ ){
        retval = PAPI_event_name_to_code( EventName[i], &events[i] );
        if( retval != PAPI_OK ) {
            fprintf( stderr, "PAPI_event_name_to_code failed\n" );
            continue;
        }
        eventCount++;
        printf( "Name %s --- Code: %#x\n", EventName[i], events[i] );
    }
 
    /* if we did not find any valid events, just report test failed. */
    if (eventCount == 0) {
        printf( "Test FAILED: no valid events found.\n");
        return 1;
    }
     
    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_create_eventset failed\n" );
     
    retval = PAPI_add_events( EventSet, events, eventCount );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_add_events failed\n" );
 #endif
    int count;
    int cuda_device;
 
    cudaGetDeviceCount( &count );
    for ( cuda_device = 0; cuda_device < count; cuda_device++ ){
        cudaSetDevice( cuda_device );
 #ifdef PAPI	
    retval = PAPI_start( EventSet );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_start failed\n" );
 #endif
 
     
//===============================================
if(strcmp(argv[1],"Add")==0){ Add<<<Db, Dg>>>(d_c); }
        else if(strcmp(argv[1],"Abs")==0){ Abs<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Bfind")==0){ Bfind<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Clz")==0){ Clz<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Cnot")==0){ Cnot<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Copysign")==0){ Copysign<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DFAdd")==0){ DFAdd<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DFDiv")==0){ DFDiv<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Div")==0){ Div<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DivU")==0){ DivU<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Ex2")==0){ Ex2<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"FastSqrt")==0){ FastSqrt<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"FDiv")==0){ FDiv<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Lg2")==0){ Lg2<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MAdd_cc")==0){ MAdd_cc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MMad_cc")==0){ MMad_cc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MSubc")==0){ MSubc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul")==0){ Mul<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul24")==0){ Mul24<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul64Hi")==0){ Mul64Hi<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Popc")==0){ Popc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rcp")==0){ Rcp<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rem")==0){ Rem<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"RemU")==0){ RemU<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rsqrt")==0){ Rsqrt<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sad")==0){ Sad<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sin")==0){ Sin<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sqrt")==0){ Sqrt<<<Db, Dg>>>(d_c); }
        else { printf("Wrong Command\n"); exit(0); }
	//mmRun();
//===============================================
cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
 
#ifdef PAPI
    retval = PAPI_stop( EventSet, values );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_stop failed\n" );
 
    for( i = 0; i < eventCount; i++ )
        printf( "On device %d: %12lld \t\t --> %s \n", cuda_device, (values[i]), EventName[i] );
#endif
}

    printf("\n");

    cudaDeviceSynchronize();

   // printf("add/sub/min/max : %d\n",((c[1]-45000000)/20000000));
 
    /* Free Device Memory */
    cudaFree(d_c);
      
    /* Free Host Memory */
    free(c);
 
    return 0;
}

// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly


#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cassert>

#include <CL/cl.h>


int main (int argc, const char** argv)
{
    // All stuff needed for sample is kept in this function body.
    // There is a couple of help macros; so they are also defined
    // directly inside this function and context dependent.

    using namespace std;

    // -----------------------------------------------------------------------
    // 1. Parse command line.

    // Default substring for platform name
    const char* required_platform_subname = "Intel";

    // Sample accepts one optional argument only, see usage information below
    if(argc > 2)
    {
        cerr << "Error: too many command line arguments.\n";
    }

    // Print usage information in case
    if(
        argc > 2 ||     // ... when wrong number of arguments is provided
        argc == 2 && (  // or user asks for help
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0
        )
        )
    {
        cout
            << "Usage: " << argv[0] << " [-h | --help | <PLATFORM>]\n\n"
            << "    -h, --help     Show this help message and exit.\n\n"
            << "    <PLATFORM>     Platform name substring to select platform.\n"
            << "                   Case sensitive. Default value is \""
            <<                     required_platform_subname << "\".\n"
            << "                   In case of multiple matches, the first matching\n"
            << "                   platform is selected.\n";

        exit(argc > 2); // return non-zero only if an error occurs
    }

    if(argc == 2)
    {
        // User provided substring for platform name selection:
        required_platform_subname = argv[1];
    }


    // -----------------------------------------------------------------------
    // 2. Define error handling strategy.

    // The following variable stores return codes for all OpenCL calls.
    // In the code it is used with CAPSBASIC_CHECK_ERRORS macro defined next.
    cl_int err = CL_SUCCESS;

    // Error handling strategy for this sample is fairly simple -- just print
    // a message and terminate the application if something goes wrong.
#define CAPSBASIC_CHECK_ERRORS(ERR)        \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    exit(1);                               \
    }


    // -----------------------------------------------------------------------
    // 3. Query for all available OpenCL platforms on the system

    cl_uint num_of_platforms = 0;
    // get total number of available platforms:
    err = clGetPlatformIDs(0, 0, &num_of_platforms);
    CAPSBASIC_CHECK_ERRORS(err);
    cout << "Number of available platforms: " << num_of_platforms << endl;

    cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
    // get IDs for all platforms:
    err = clGetPlatformIDs(num_of_platforms, platforms, 0);
    CAPSBASIC_CHECK_ERRORS(err);


    // -----------------------------------------------------------------------
    // 4. List all platforms and select one.
    // We use platform name to select needed platform.

    cl_uint selected_platform_index = num_of_platforms;

    cout << "Platform names:\n";

    for(cl_uint i = 0; i < num_of_platforms; ++i)
    {
        // Get the length for the i-th platform name
        size_t platform_name_length = 0;
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
            );
        CAPSBASIC_CHECK_ERRORS(err);

        // Get the name itself for the i-th platform
        char* platform_name = new char[platform_name_length];
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            platform_name_length,
            platform_name,
            0
            );
        CAPSBASIC_CHECK_ERRORS(err);

        cout << "    [" << i << "] " << platform_name;

        // decide if this i-th platform is what we are looking for
        // we select the first one matched skipping the next one if any
        if(
            strstr(platform_name, required_platform_subname) &&
            selected_platform_index == num_of_platforms // have not selected yet
            )
        {
            cout << " [Selected]";
            selected_platform_index = i;
            // do not stop here, just see all available platforms
        }

        cout << endl;
        delete [] platform_name;
    }

    if(selected_platform_index == num_of_platforms)
    {
        cerr
            << "There is no found platform with name containing \""
            << required_platform_subname << "\" as a substring.\n";
        return 1;
    }

    cl_platform_id platform = platforms[selected_platform_index];


    // -----------------------------------------------------------------------
    // 5. Let us see how many devices of each type are provided for the
    // selected platform.

    // Use the following handy array to
    // store all device types of your interest. The array helps to build simple
    // loop queries in the code below.

    struct
    {
        cl_device_type type;
        const char* name;
        cl_uint count;
    }
    devices[] =
    {
        { CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", 0 },
        { CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", 0 },
        { CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", 0 }
    };

    const int NUM_OF_DEVICE_TYPES = sizeof(devices)/sizeof(devices[0]);

    cout << "Number of devices available for each type:\n";

    // Now iterate over all device types picked above and
    // initialize num_of_devices
    for(int i = 0; i < NUM_OF_DEVICE_TYPES; ++i)
    {
        err = clGetDeviceIDs(
            platform,
            devices[i].type,
            0,
            0,
            &devices[i].count
            );

        if(CL_DEVICE_NOT_FOUND == err)
        {
            // that's OK to fall here, because not all types of devices, which
            // you query for may be available for a particular system
            devices[i].count = 0;
            err = CL_SUCCESS;
        }

        CAPSBASIC_CHECK_ERRORS(err);

        cout
            << "    " << devices[i].name << ": "
            << devices[i].count << endl;
    }


    // -----------------------------------------------------------------------
    // 6. Now get a piece of useful capabilities information for each device.
    // Group information by device type

    cout << "\n*** Detailed information for each device ***\n";

    for(int type_index = 0; type_index < NUM_OF_DEVICE_TYPES; ++type_index)
    {
        cl_uint cur_num_of_devices = devices[type_index].count;

        if(cur_num_of_devices == 0)
        {
            // there is no devices of this type; move to the next type
            continue;
        }

        // Retrieve a list of device IDs with type selected by type_index
        cl_device_id* devices_of_type = new cl_device_id[cur_num_of_devices];
        err = clGetDeviceIDs(
            platform,
            devices[type_index].type,
            cur_num_of_devices,
            devices_of_type,
            0
            );
        CAPSBASIC_CHECK_ERRORS(err);

        // Iterate over all devices of the current device type
        for(
            cl_uint device_index = 0;
            device_index < cur_num_of_devices;
        ++device_index
            )
        {
            cout
                << "\n"
                << devices[type_index].name
                << "[" << device_index << "]\n";

            cl_device_id device = devices_of_type[device_index];

            // To enumerate capabilities information, use two help
            // macros: one to print string information and another one to
            // print numeric information. Both these macros use clGetDeviceInfo
            // to retrieve required caps, and defined below:

#define OCLBASIC_PRINT_TEXT_PROPERTY(NAME)                       \
            {                                                    \
            /* When we query for string properties, first we */  \
            /* need to get string length:                    */  \
            size_t property_length = 0;                          \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            0,                                                   \
            0,                                                   \
            &property_length                                     \
            );                                                   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            /* Then allocate buffer. No need to add 1 symbol */  \
            /* to store terminating zero; OpenCL takes care  */  \
            /* about it:                                     */  \
            char* property_value = new char[property_length];    \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            property_length,                                     \
            property_value,                                      \
            0                                                    \
            );                                                   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            cout                                                 \
            << "    " << #NAME << ": "                           \
            << property_value << endl;                           \
            delete [] property_value;                            \
            }

#define OCLBASIC_PRINT_NUMERIC_PROPERTY(NAME, TYPE)              \
            {                                                    \
            TYPE property_value;                                 \
            size_t property_length = 0;                          \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            sizeof(property_value),                              \
            &property_value,                                     \
            &property_length                                     \
            );                                                   \
            assert(property_length == sizeof(property_value));   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            cout                                                 \
            << "    " << #NAME << ": "                           \
            << property_value << endl;                           \
            }


            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_NAME);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_AVAILABLE, cl_bool);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VENDOR);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_PROFILE);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DRIVER_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_OPENCL_C_VERSION);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ADDRESS_BITS, cl_uint);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_IMAGE_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool);

            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_EXTENSIONS);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint);
        }

        delete [] devices_of_type;
    }


    // -----------------------------------------------------------------------
    // Final clean up

    delete [] platforms;

    return 0;
}

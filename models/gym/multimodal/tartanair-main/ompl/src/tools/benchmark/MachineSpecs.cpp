/** * @author: AirLab / Field Robotics Center
 *
 * @attention Copyright (C) 2016
 * @attention Carnegie Mellon University
 * @attention All rights reserved
 *
 * @attention LIMITED RIGHTS:
 * @attention The US Government is granted Limited Rights to this Data.
 *            Use, duplication, or disclosure is subject to the
 *            restrictions as stated in Agreement AFS12-1642.
 */
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: The Internet */

#include "ompl/tools/benchmark/MachineSpecs.h"
#include "ompl/util/Console.h"

/// @cond IGNORE

#if defined _WIN32

// Windows 2000 or newer
#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#include <psapi.h>

ompl::machine::MemUsage_t getProcessMemoryUsageAux(void)
{
    HANDLE hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    hProcess = OpenProcess(  PROCESS_QUERY_INFORMATION |
                             PROCESS_VM_READ,
                             false,
                             GetCurrentProcessId() );

    ompl::machine::MemUsage_t result = 0;

    if (NULL != hProcess)
    {
        if ( GetProcessMemoryInfo( hProcess, &pmc, sizeof(pmc)) )
            result = pmc.WorkingSetSize;
        CloseHandle( hProcess );
    }

    return result;
}

#else
#if defined __APPLE__

// Mac OS 10.2 or newer
#include <mach/mach_init.h>
#include <mach/task.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <cstring>
#include <unistd.h>

ompl::machine::MemUsage_t getProcessMemoryUsageAux(void)
{

    task_basic_info         info;
    kern_return_t           rval = 0;
    mach_port_t             task = mach_task_self();
    mach_msg_type_number_t  tcnt = TASK_BASIC_INFO_COUNT;
    task_info_t             tptr = (task_info_t) &info;

    memset(&info, 0, sizeof(info));

    rval = task_info(task, TASK_BASIC_INFO, tptr, &tcnt);
    if (!(rval == KERN_SUCCESS)) return 0;
    return info.resident_size;
}

#else
#if defined _POSIX_VERSION || defined _POSIX2_VERSION || defined __linux__
// we need a posix compliant os that exposes /proc/self/stat

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>

ompl::machine::MemUsage_t getProcessMemoryUsageAux(void)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   if (stat_stream.good() && !stat_stream.eof())
   {
       // dummy vars for leading entries in stat that we don't care about
       //
       string pid, comm, state, ppid, pgrp, session, tty_nr;
       string tpgid, flags, minflt, cminflt, majflt, cmajflt;
       string utime, stime, cutime, cstime, priority, nice;
       string O, itrealvalue, starttime;


       // the two fields we want
       //
       unsigned long vsize;
       long rss;

       stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                   >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                   >> utime >> stime >> cutime >> cstime >> priority >> nice
                   >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

       ompl::machine::MemUsage_t page_size = sysconf(_SC_PAGE_SIZE);
       return rss * page_size;
   }
   return 0;
}

#else
// if we have no idea what to do, we return 0
ompl::machine::MemUsage_t getProcessMemoryUsageAux(void)
{
    return 0;
}
#endif // posix
#endif // apple
#endif // windows

ompl::machine::MemUsage_t ompl::machine::getProcessMemoryUsage(void)
{
    MemUsage_t result = getProcessMemoryUsageAux();
    if (result == 0)
    {
        OMPL_WARN("Unable to get memory usage");
    }
    return result;
}

std::string ompl::machine::getHostname(void)
{
    static const int BUF_SIZE = 1024;
    char buffer[BUF_SIZE];
    int len = gethostname(buffer, sizeof(buffer));
    if (len != 0)
        return std::string();
    else
    {
        buffer[BUF_SIZE - 1] = '\0';
        return std::string(buffer);
    }
}

/// @endcond

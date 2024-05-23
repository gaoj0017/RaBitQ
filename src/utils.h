#pragma once
#include <chrono>
#include <queue>
#include <unordered_set>
#include <limits>	

#ifndef WIN32
#include<sys/resource.h>
#endif

typedef std::pair<float, uint32_t> Result;
typedef std::priority_queue<Result> ResultHeap; 
	
namespace Detail
{
	double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
	{
		return curr == prev
			? curr
			: sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
	}
}

/*
* Constexpr version of the square root
* Return value:
*	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
double constexpr const_sqrt(double x)
{
	return x >= 0 && x < std::numeric_limits<double>::infinity()
		? Detail::sqrtNewtonRaphson(x, x, 0)
		: std::numeric_limits<double>::quiet_NaN();
}

void print_binary(uint64_t v){
    for(int i=0;i<64;i++){
        std::cerr << ((v >> (63 - i)) & 1);
    }
}

void print_binary(uint8_t v){
    for(int i=0;i<8;i++){
        std::cerr << ((v >> (7 - i)) & 1);
    }
}

inline uint32_t reverseBits(uint32_t n) {
    n = (n >> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
    n = (n >> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
    n = (n >> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
    n = (n >> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
    n = (n >> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
    return n;
}

ResultHeap getGroundtruth(const Matrix<float> & X, const Matrix<float> & Q, size_t query,
                        unsigned* groundtruth, size_t k){
    ResultHeap ret;
    for(int i=0;i<k;i++){
        unsigned gt = groundtruth[i];
        ret.push(std::make_pair(Q.dist(query, X, gt), gt));
    }
    return ret;
}

float getRatio(int q, const Matrix<float> &Q, const Matrix<float> &X, const Matrix<unsigned> &G, ResultHeap KNNs){
    ResultHeap gt;
    int k = KNNs.size();
    for(int i=0;i<k;i++){
        gt.emplace(Q.dist(q, X, G.data[q * G.d + i]), G.data[q * G.d + i]);
    }
    long double ret=0;
    int valid_k=0;
    while(gt.size()){
        if(gt.top().first > 1e-5){
            ret += std::sqrt(KNNs.top().first / gt.top().first);
            valid_k ++;
        }
        gt.pop();
        KNNs.pop();
    }
    if(valid_k == 0) return 1.0 * k;
    return ret / valid_k * k;
}

int getRecall(ResultHeap & result, ResultHeap & gt){
    int correct=0;
    
    std::unordered_set<unsigned> g;
    int ret = 0;

    while (gt.size()) {
        g.insert(gt.top().second);
        //std::cerr << "ID - " << gt.top().second << " dist - " << gt.top().first << std::endl;
        gt.pop();
    }

    while (result.size()) {
        //std::cerr << "ID - " << result.top().second << " dist - " << result.top().first << std::endl;
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }
    
    return ret;
}

#ifndef WIN32
void GetCurTime( rusage* curTime)
{
	int ret = getrusage( RUSAGE_SELF, curTime);
	if( ret != 0)
	{
		fprintf( stderr, "The running time info couldn't be collected successfully.\n");
		//FreeData( 2);
		exit( 0);
	}
}

/*
* GetTime is used to get the 'float' format time from the start and end rusage structure.
* 
* @Param timeStart, timeEnd indicate the two time points.
* @Param userTime, sysTime get back the time information.
* 
* @Return void.
*/
void GetTime( struct rusage* timeStart, struct rusage* timeEnd, float* userTime, float* sysTime)
{
	(*userTime) = ((float)(timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) + 
		((float)(timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-6;
	(*sysTime) = ((float)(timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) +
		((float)(timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-6;
}

#endif

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}
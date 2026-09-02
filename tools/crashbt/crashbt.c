#define _GNU_SOURCE
#include <signal.h>
#include <execinfo.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
static void handler(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    int fd = open("/home/zhaosiying/zcode-lane/artifacts/hybrid-cta-capsule-v2/mini-engine/crashbt.log",
                  O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) _exit(128 + sig);
    void *bt[64];
    int n = backtrace(bt, 64);
    dprintf(fd, "===CRASHBT sig=%d fault_addr=%p pid=%d===\n",
            sig, si->si_addr, (int)getpid());
    backtrace_symbols_fd(bt, n, fd);
    close(fd);
    _exit(128 + sig);
}
__attribute__((constructor))
static void crashbt_init(void) {
    struct sigaction sa;
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, 0);
    sigaction(SIGABRT, &sa, 0);
}
